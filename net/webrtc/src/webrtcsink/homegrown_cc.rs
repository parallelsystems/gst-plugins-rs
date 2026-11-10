// SPDX-License-Identifier: MPL-2.0

use gst::{
    glib::{self, value::FromValue},
    prelude::*,
};
use once_cell::sync::Lazy;

use super::imp::VideoEncoder;

static CAT: Lazy<gst::DebugCategory> = Lazy::new(|| {
    gst::DebugCategory::new(
        "webrtcsink-homegrowncc",
        gst::DebugColorFlags::empty(),
        Some("WebRTC sink"),
    )
});

/// The TWCC stats exposed by rtpsession cover a (hard-coded) 300ms window.
/// Only act on them once per window, so that the same packets (e.g. one
/// keyframe burst) don't drive several consecutive bitrate changes.
const DELAY_CONTROL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(300);

/// How long the delay-based overuse signal must hold before the bitrate is
/// decreased. A single large frame (keyframe) is spread out by any bottleneck
/// and looks like overuse for as long as it stays in the stats window
/// (~300ms); this must be longer than that for low framerate streams with
/// short GOPs not to trigger a decrease on every keyframe.
const OVERUSE_PERSISTENCE: std::time::Duration = std::time::Duration::from_millis(500);

/// Multiplicative increase factor, applied once per DELAY_CONTROL_INTERVAL
const DELAY_MULTIPLICATIVE_INCREASE: f64 = 1.06;

/// Total arrival skew (receive duration - send duration) accumulated over
/// the stats window above which the delay controller signals overuse,
/// i.e. 10% of the 300ms window
const MAX_WINDOW_SKEW_NS: i64 = 30_000_000;

/// Time constant for the smoothing of the loss percentage. With few packets
/// per stats window (low framerate / low bitrate), a single lost packet is
/// worth several percents of loss, smoothing avoids overreacting to it.
const LOSS_SMOOTHING_TIME_CONSTANT_S: f64 = 1.0;

/// Framerate assumed when the actual one is unknown
const DEFAULT_FRAMERATE: f64 = 30.;

#[derive(Debug)]
enum IncreaseType {
    /// Increase bitrate by value
    Additive(f64),
    /// Increase bitrate by factor
    Multiplicative(f64),
}

#[derive(Debug, Clone, Copy)]
enum ControllerType {
    // Running the "delay-based controller"
    Delay,
    // Running the "loss based controller"
    Loss,
}

#[derive(Debug)]
enum CongestionControlOp {
    /// Don't update target bitrate
    Hold,
    /// Decrease target bitrate
    Decrease {
        factor: f64,
        #[allow(dead_code)]
        reason: String, // for Debug
    },
    /// Increase target bitrate, either additively or multiplicatively
    Increase(IncreaseType),
}

fn lookup_twcc_stats(stats: &gst::StructureRef) -> Option<gst::Structure> {
    for (_, field_value) in stats {
        if let Ok(s) = field_value.get::<gst::Structure>() {
            if let Ok(type_) = s.get::<gst_webrtc::WebRTCStatsType>("type") {
                if (type_ == gst_webrtc::WebRTCStatsType::Transport
                    || type_ == gst_webrtc::WebRTCStatsType::CandidatePair)
                    && s.has_field("gst-twcc-stats")
                {
                    return Some(s.get::<gst::Structure>("gst-twcc-stats").unwrap());
                }
            }
        }
    }

    None
}

pub struct CongestionController {
    /// Note: The target bitrate applied is the min of
    /// target_bitrate_on_delay and target_bitrate_on_loss
    ///
    /// Bitrate target based on delay factor for all video streams.
    /// Hasn't been tested with multiple video streams, but
    /// current design is simply to divide bitrate equally.
    pub target_bitrate_on_delay: i32,

    /// Bitrate target based on loss for all video streams.
    pub target_bitrate_on_loss: i32,

    /// Exponential moving average, updated when bitrate is
    /// decreased, discarded when increased again past last
    /// congestion window. Smoothing factor hardcoded.
    bitrate_ema: Option<f64>,
    /// Exponentially weighted moving variance, recursively
    /// updated along with bitrate_ema. sqrt'd to obtain standard
    /// deviation, used to determine whether to increase bitrate
    /// additively or multiplicatively
    bitrate_emvar: f64,
    /// Used in additive mode to track last control time, influences
    /// calculation of added value according to gcc section 5.5
    last_update_time: Option<std::time::Instant>,
    /// When the delay-based overuse signal was first observed, reset
    /// as soon as a sample shows no overuse
    overuse_since: Option<std::time::Instant>,
    /// Smoothed loss percentage and the time it was last updated
    loss_pct_ema: Option<(f64, std::time::Instant)>,
    /// For logging purposes
    peer_id: String,

    min_bitrate: u32,
    max_bitrate: u32,
    do_fec: bool,
}

impl CongestionController {
    pub fn new(peer_id: &str, min_bitrate: u32, max_bitrate: u32, do_fec: bool) -> Self {
        Self {
            target_bitrate_on_delay: 0,
            target_bitrate_on_loss: 0,
            bitrate_ema: None,
            bitrate_emvar: 0.,
            last_update_time: None,
            overuse_since: None,
            loss_pct_ema: None,
            peer_id: peer_id.to_string(),
            min_bitrate,
            max_bitrate,
            do_fec,
        }
    }

    fn update_delay(
        &mut self,
        element: &super::BaseWebRTCSink,
        twcc_stats: &gst::StructureRef,
        rtt: f64,
        framerate: f64,
    ) -> Option<CongestionControlOp> {
        let target_bitrate = f64::min(
            self.target_bitrate_on_delay as f64,
            self.target_bitrate_on_loss as f64,
        );
        // Unwrap, all those fields must be there or there's been an API
        // break, which qualifies as programming error
        let bitrate_sent = twcc_stats.get::<u32>("bitrate-sent").unwrap();
        let bitrate_recv = twcc_stats.get::<u32>("bitrate-recv").unwrap();
        let packets_sent = twcc_stats.get::<u32>("packets-sent").unwrap();
        let delta_of_delta = twcc_stats.get::<i64>("avg-delta-of-delta").unwrap();

        let sent_minus_received = bitrate_sent.saturating_sub(bitrate_recv);

        // Relative to what was actually sent in the window rather than to the
        // target: the sent bitrate is well above target whenever the window
        // contains a keyframe, which would otherwise inflate the factor.
        let delay_factor = if bitrate_sent > 0 {
            sent_minus_received as f64 / bitrate_sent as f64
        } else {
            0.
        };

        // avg-delta-of-delta is a per-packet mean, which makes any fixed
        // threshold on it depend on the packet rate. Consider the skew
        // accumulated over the whole window instead.
        let window_skew = delta_of_delta.saturating_mul(packets_sent as i64);

        gst::trace!(
            CAT,
            obj = element,
            "consumer {}: considering stats {}",
            self.peer_id,
            twcc_stats
        );

        let overuse = if delay_factor > 0.1 {
            Some(if delay_factor < 0.64 {
                (0.96, format!("low delay factor {delay_factor}"))
            } else {
                (
                    delay_factor.sqrt().sqrt().clamp(0.8, 0.96),
                    format!("High delay factor {delay_factor}"),
                )
            })
        } else if window_skew > MAX_WINDOW_SKEW_NS {
            Some((0.97, format!("High window skew: {window_skew}ns")))
        } else {
            None
        };

        let now = std::time::Instant::now();

        // Every sample is used to track how long overuse has been signaled,
        // even those we won't act upon
        if overuse.is_some() {
            self.overuse_since.get_or_insert(now);
        } else {
            self.overuse_since = None;
        }

        if let Some(last_update_time) = self.last_update_time {
            // Stats are polled on a 100ms timer, leave some slack for the
            // timer's jitter so that we run on every third tick
            if now.duration_since(last_update_time) + std::time::Duration::from_millis(20)
                < DELAY_CONTROL_INTERVAL
            {
                return None;
            }
        }
        let last_update_time = self.last_update_time.replace(now);

        let op = if let Some((factor, reason)) = overuse {
            let overuse_duration = self
                .overuse_since
                .map_or(std::time::Duration::ZERO, |since| now.duration_since(since));

            if overuse_duration >= OVERUSE_PERSISTENCE {
                CongestionControlOp::Decrease { factor, reason }
            } else {
                gst::trace!(
                    CAT,
                    obj = element,
                    "consumer {}: overuse signaled for {:?} ({}), holding",
                    self.peer_id,
                    overuse_duration,
                    reason
                );
                CongestionControlOp::Hold
            }
        } else {
            CongestionControlOp::Increase(if let Some(ema) = self.bitrate_ema {
                let bitrate_stdev = self.bitrate_emvar.sqrt();

                gst::trace!(
                    CAT,
                    obj = element,
                    "consumer {}: Old bitrate: {}, ema: {}, stddev: {}",
                    self.peer_id,
                    target_bitrate,
                    ema,
                    bitrate_stdev,
                );

                // gcc section 5.5 advises 3 standard deviations, but experiments
                // have shown this to be too low, probably related to the rest of
                // homegrown algorithm not implementing gcc, revisit when implementing
                // the rest of the RFC
                if target_bitrate < ema - 7. * bitrate_stdev {
                    gst::trace!(
                        CAT,
                        obj = element,
                        "consumer {}: below last congestion window",
                        self.peer_id
                    );
                    /* Multiplicative increase */
                    IncreaseType::Multiplicative(DELAY_MULTIPLICATIVE_INCREASE)
                } else if target_bitrate > ema + 7. * bitrate_stdev {
                    gst::trace!(
                        CAT,
                        obj = element,
                        "consumer {}: above last congestion window",
                        self.peer_id
                    );
                    /* We have gone past our last estimated max bandwidth
                     * network situation may have changed, go back to
                     * multiplicative increase
                     */
                    self.bitrate_ema.take();
                    IncreaseType::Multiplicative(DELAY_MULTIPLICATIVE_INCREASE)
                } else {
                    let rtt_ms = rtt * 1000.;
                    let response_time_ms = 100. + rtt_ms;
                    let time_since_last_update_ms = match last_update_time {
                        None => 0.,
                        Some(instant) => now.duration_since(instant).as_millis() as f64,
                    };
                    // gcc section 5.5 advises 0.95 as the smoothing factor, but that
                    // seems intuitively much too low, granting disproportionate importance
                    // to the last measurement. 0.5 seems plenty enough, I don't have maths
                    // to back that up though :)
                    let alpha = 0.5 * f64::min(time_since_last_update_ms / response_time_ms, 1.0);
                    let bits_per_frame = target_bitrate / framerate;
                    let packets_per_frame = f64::ceil(bits_per_frame / (1200. * 8.));
                    let avg_packet_size_bits = bits_per_frame / packets_per_frame;

                    gst::trace!(
                        CAT,
                        obj = element,
                        "consumer {}: still in last congestion window",
                        self.peer_id,
                    );

                    /* Additive increase */
                    IncreaseType::Additive(f64::max(1000., alpha * avg_packet_size_bits))
                }
            } else {
                /* Multiplicative increase */
                gst::trace!(
                    CAT,
                    obj = element,
                    "consumer {}: outside congestion window",
                    self.peer_id
                );
                IncreaseType::Multiplicative(DELAY_MULTIPLICATIVE_INCREASE)
            })
        };

        Some(op)
    }

    fn clamp_bitrate(&mut self, bitrate: i32, n_encoders: i32, controller_type: ControllerType) {
        match controller_type {
            ControllerType::Loss => {
                self.target_bitrate_on_loss = bitrate.clamp(
                    self.min_bitrate as i32 * n_encoders,
                    self.max_bitrate as i32 * n_encoders,
                )
            }

            ControllerType::Delay => {
                self.target_bitrate_on_delay = bitrate.clamp(
                    self.min_bitrate as i32 * n_encoders,
                    self.max_bitrate as i32 * n_encoders,
                )
            }
        }
    }

    fn get_remote_inbound_stats(&self, stats: &gst::StructureRef) -> Vec<gst::Structure> {
        let mut inbound_rtp_stats: Vec<gst::Structure> = Default::default();
        for (_, field_value) in stats {
            if let Ok(s) = field_value.get::<gst::Structure>() {
                if let Ok(type_) = s.get::<gst_webrtc::WebRTCStatsType>("type") {
                    if type_ == gst_webrtc::WebRTCStatsType::RemoteInboundRtp {
                        inbound_rtp_stats.push(s);
                    }
                }
            }
        }

        inbound_rtp_stats
    }

    fn lookup_rtt(&self, stats: &gst::StructureRef) -> f64 {
        let inbound_rtp_stats = self.get_remote_inbound_stats(stats);
        let mut rtt = 0.;
        let mut n_rtts = 0u64;
        for inbound_stat in &inbound_rtp_stats {
            if let Err(err) = (|| -> Result<(), gst::structure::GetError<<<f64 as FromValue>::Checker as glib::value::ValueTypeChecker>::Error>> {
                rtt += inbound_stat.get::<f64>("round-trip-time")?;
                n_rtts += 1;

                Ok(())
            })() {
                gst::debug!(CAT, "{:?}", err);
            }
        }

        rtt /= f64::max(1., n_rtts as f64);

        gst::log!(CAT, "Round trip time: {}", rtt);

        rtt
    }

    pub fn loss_control(
        &mut self,
        element: &super::BaseWebRTCSink,
        stats: &gst::StructureRef,
        encoders: &mut [VideoEncoder],
    ) {
        let raw_loss_percentage = stats.get::<f64>("packet-loss-pct").unwrap();

        // No packet in the window (0 / 0), nothing to learn from it
        if !raw_loss_percentage.is_finite() {
            return;
        }

        // Time-weighted exponential smoothing: feedbacks don't come at a
        // fixed rate and each covers an overlapping 300ms window.
        let now = std::time::Instant::now();
        let loss_percentage = match self.loss_pct_ema {
            Some((ema, last)) => {
                let dt = now.duration_since(last).as_secs_f64();
                let alpha = 1. - f64::exp(-dt / LOSS_SMOOTHING_TIME_CONSTANT_S);
                ema + alpha * (raw_loss_percentage - ema)
            }
            None => raw_loss_percentage,
        };
        self.loss_pct_ema = Some((loss_percentage, now));

        gst::trace!(
            CAT,
            obj = element,
            "consumer {}: loss {raw_loss_percentage}%, smoothed {loss_percentage}%",
            self.peer_id,
        );

        self.apply_control_op(
            element,
            encoders,
            if loss_percentage > 10. {
                CongestionControlOp::Decrease {
                    factor: ((100. - (0.5 * loss_percentage)) / 100.).clamp(0.7, 0.98),
                    reason: format!("High loss: {loss_percentage}"),
                }
            } else if loss_percentage > 2. {
                CongestionControlOp::Hold
            } else {
                CongestionControlOp::Increase(IncreaseType::Multiplicative(1.05))
            },
            ControllerType::Loss,
        );
    }

    pub fn delay_control(
        &mut self,
        element: &super::BaseWebRTCSink,
        stats: &gst::StructureRef,
        encoders: &mut [VideoEncoder],
    ) {
        if let Some(twcc_stats) = lookup_twcc_stats(stats) {
            let framerate = encoders
                .first()
                .and_then(|encoder| encoder.framerate())
                .unwrap_or(DEFAULT_FRAMERATE);
            let rtt = self.lookup_rtt(stats);

            // None when it is too early to act on the stats again
            if let Some(op) = self.update_delay(element, &twcc_stats, rtt, framerate) {
                self.apply_control_op(element, encoders, op, ControllerType::Delay);
            }
        }
    }

    fn apply_control_op(
        &mut self,
        element: &super::BaseWebRTCSink,
        encoders: &mut [VideoEncoder],
        control_op: CongestionControlOp,
        controller_type: ControllerType,
    ) {
        gst::trace!(
            CAT,
            obj = element,
            "consumer {}: applying congestion control operation {:?}",
            self.peer_id,
            control_op
        );

        let n_encoders = encoders.len() as i32;
        let prev_bitrate = i32::min(self.target_bitrate_on_delay, self.target_bitrate_on_loss);
        match &control_op {
            CongestionControlOp::Hold => {}
            CongestionControlOp::Increase(IncreaseType::Additive(value)) => {
                self.clamp_bitrate(
                    self.target_bitrate_on_delay + *value as i32,
                    n_encoders,
                    controller_type,
                );
            }
            CongestionControlOp::Increase(IncreaseType::Multiplicative(factor)) => {
                self.clamp_bitrate(
                    (self.target_bitrate_on_delay as f64 * factor) as i32,
                    n_encoders,
                    controller_type,
                );
            }
            CongestionControlOp::Decrease { factor, .. } => {
                self.clamp_bitrate(
                    (self.target_bitrate_on_delay as f64 * factor) as i32,
                    n_encoders,
                    controller_type,
                );

                if let ControllerType::Delay = controller_type {
                    // Smoothing factor
                    let alpha = 0.75;
                    if let Some(ema) = self.bitrate_ema {
                        let sigma: f64 = (self.target_bitrate_on_delay as f64) - ema;
                        self.bitrate_ema = Some(ema + (alpha * sigma));
                        self.bitrate_emvar =
                            (1. - alpha) * (self.bitrate_emvar + alpha * sigma.powi(2));
                    } else {
                        self.bitrate_ema = Some(self.target_bitrate_on_delay as f64);
                        self.bitrate_emvar = 0.;
                    }
                }
            }
        }

        let target_bitrate =
            i32::min(self.target_bitrate_on_delay, self.target_bitrate_on_loss).clamp(
                self.min_bitrate as i32 * n_encoders,
                self.max_bitrate as i32 * n_encoders,
            ) / n_encoders;

        if target_bitrate != prev_bitrate {
            gst::info!(
                CAT,
                "{:?} {} => {} | on delay {} - on loss {} | min {} - max {}",
                control_op,
                human_bytes::human_bytes(prev_bitrate),
                human_bytes::human_bytes(target_bitrate),
                human_bytes::human_bytes(self.target_bitrate_on_delay),
                human_bytes::human_bytes(self.target_bitrate_on_loss),
                human_bytes::human_bytes(self.min_bitrate),
                human_bytes::human_bytes(self.max_bitrate),
            );
        
            let fec_ratio = if self.do_fec {
                if target_bitrate <= 2000000 || self.max_bitrate <= 2000000 {
                    0f64
                } else {
                    (target_bitrate as f64 - 2000000f64) / (self.max_bitrate as f64 - 2000000f64)
                }
            } else {
                0.0
            };

            let fec_percentage = (fec_ratio * 50f64) as u32;

            for encoder in encoders.iter_mut() {
                if encoder.set_bitrate(element, target_bitrate).is_ok() {
                    encoder
                        .transceiver
                        .set_property("fec-percentage", fec_percentage);
                }
            }

            // Emit a signal through the parent `webrtcsink`
            element.emit_by_name::<()>("cc-bitrate-changed", &[&(self.target_bitrate_on_delay as u32), &(self.target_bitrate_on_loss as u32), &(target_bitrate as u32)]);
        }
    }
}
