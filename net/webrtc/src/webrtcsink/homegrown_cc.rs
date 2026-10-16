// SPDX-License-Identifier: MPL-2.0

use gst::{
    glib::{self, value::FromValue},
    prelude::*,
};
use once_cell::sync::Lazy;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

use super::imp::VideoEncoder;

/// Length of the sliding window over which the delay signals are averaged
/// before being compared against their thresholds.
///
/// This should be at least one GOP long. The TWCC stats are computed over a
/// short window, and the sender doesn't pace, so every packet of a frame gets
/// roughly the same send timestamp while the receiver sees them spread out by
/// the time it takes the frame to drain through the bottleneck. Whenever the
/// newest frame in that window is a keyframe, bitrate-recv reads noticeably
/// lower than bitrate-sent even on an idle link. Without averaging, that
/// shows up as a spurious decrease once per keyframe. Averaging over a full
/// GOP weighs the keyframe tick against the rest of the GOP.
const DELAY_SIGNAL_WINDOW: Duration = Duration::from_millis(500);

/// Averaged (sent - recv) / sent above which the delay controller decreases
const DELAY_FACTOR_THRESHOLD: f64 = 0.1;

/// Averaged avg-delta-of-delta (ns) above which the delay controller decreases
const DELTA_OF_DELTA_THRESHOLD_NS: f64 = 1_000_000.;

/// Sliding-window average of the delay controller's input signals.
#[derive(Debug, Default)]
struct DelaySignals {
    /// (sample time, signed (sent - recv) / sent, avg-delta-of-delta in ns)
    samples: VecDeque<(Instant, f64, f64)>,
    /// When collection started since the last reset
    since: Option<Instant>,
}

impl DelaySignals {
    fn push(&mut self, now: Instant, rel_diff: f64, delta_of_delta: f64) {
        self.since.get_or_insert(now);
        self.samples.push_back((now, rel_diff, delta_of_delta));
        while let Some(&(t, ..)) = self.samples.front() {
            if now.duration_since(t) >= DELAY_SIGNAL_WINDOW {
                self.samples.pop_front();
            } else {
                break;
            }
        }
    }

    /// Mean (delay_factor, delta_of_delta) over the window, or None until
    /// samples have been collected for a full window since the last reset.
    fn averages(&self, now: Instant) -> Option<(f64, f64)> {
        let since = self.since?;
        if now.duration_since(since) < DELAY_SIGNAL_WINDOW || self.samples.is_empty() {
            return None;
        }

        let n = self.samples.len() as f64;
        let (rel_sum, dod_sum) = self
            .samples
            .iter()
            .fold((0., 0.), |(r, d), &(_, rel, dod)| (r + rel, d + dod));

        Some((rel_sum / n, dod_sum / n))
    }

    /// Drop all samples, e.g. after a decrease: samples taken at the old
    /// bitrate say nothing about the new one, and keeping them would make
    /// the controller cut again on every tick until they age out.
    fn reset(&mut self) {
        self.samples.clear();
        self.since = None;
    }
}

static CAT: Lazy<gst::DebugCategory> = Lazy::new(|| {
    gst::DebugCategory::new(
        "webrtcsink-homegrowncc",
        gst::DebugColorFlags::empty(),
        Some("WebRTC sink"),
    )
});

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
    /// Delay signals averaged over DELAY_SIGNAL_WINDOW before thresholding
    delay_signals: DelaySignals,
    /// For logging purposes
    peer_id: String,

    min_bitrate: u32,
    max_bitrate: u32,
    do_fec: bool,
    do_apply: bool,
}

impl CongestionController {
    pub fn new(
        peer_id: &str,
        min_bitrate: u32,
        max_bitrate: u32,
        do_fec: bool,
        do_apply: bool,
    ) -> Self {
        Self {
            target_bitrate_on_delay: 0,
            target_bitrate_on_loss: 0,
            bitrate_ema: None,
            bitrate_emvar: 0.,
            last_update_time: None,
            delay_signals: DelaySignals::default(),
            peer_id: peer_id.to_string(),
            min_bitrate,
            max_bitrate,
            do_fec,
            do_apply,
        }
    }

    fn update_delay(
        &mut self,
        element: &super::BaseWebRTCSink,
        twcc_stats: &gst::StructureRef,
        rtt: f64,
    ) -> CongestionControlOp {
        let target_bitrate = f64::min(
            self.target_bitrate_on_delay as f64,
            self.target_bitrate_on_loss as f64,
        );
        // Unwrap, all those fields must be there or there's been an API
        // break, which qualifies as programming error
        let bitrate_sent = twcc_stats.get::<u32>("bitrate-sent").unwrap();
        let bitrate_recv = twcc_stats.get::<u32>("bitrate-recv").unwrap();
        let delta_of_delta = twcc_stats.get::<i64>("avg-delta-of-delta").unwrap();

        let now = Instant::now();
        let last_update_time = self.last_update_time.replace(now);

        gst::trace!(
            CAT,
            obj = element,
            "consumer {}: considering stats {}",
            self.peer_id,
            twcc_stats
        );

        // Signed on purpose: burst-drain artifacts push recv both above and
        // below sent, and only a sustained excess of sent over recv means
        // congestion. Clamping each sample to >= 0 (saturating_sub) before
        // averaging would turn that zero-mean noise into a positive bias.
        if bitrate_sent > 0 {
            // @viotovmp - normalized by `bitrate_sent` instead of `target_bitrate`
            let rel_diff =
                (f64::from(bitrate_sent) - f64::from(bitrate_recv)) / f64::from(bitrate_sent);
            self.delay_signals
                .push(now, rel_diff, delta_of_delta as f64);
        }

        let Some((delay_factor, delta_of_delta)) = self.delay_signals.averages(now) else {
            gst::trace!(
                CAT,
                obj = element,
                "consumer {}: collecting delay samples, holding",
                self.peer_id
            );
            return CongestionControlOp::Hold;
        };

        gst::trace!(
            CAT,
            obj = element,
            "consumer {}: averaged delay factor {:.4}, delta-of-delta {:.0} ns over {} samples",
            self.peer_id,
            delay_factor,
            delta_of_delta,
            self.delay_signals.samples.len()
        );

        if delay_factor > DELAY_FACTOR_THRESHOLD {
            let (factor, reason) = if delay_factor < 0.64 {
                (0.96, format!("low delay factor {delay_factor}"))
            } else {
                (
                    delay_factor.sqrt().sqrt().clamp(0.8, 0.96),
                    format!("High delay factor {delay_factor}"),
                )
            };

            self.delay_signals.reset();
            CongestionControlOp::Decrease { factor, reason }
        } else if delta_of_delta > DELTA_OF_DELTA_THRESHOLD_NS {
            self.delay_signals.reset();
            CongestionControlOp::Decrease {
                factor: 0.97,
                reason: format!("High delta: {delta_of_delta}"),
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
                    IncreaseType::Multiplicative(1.03)
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
                    IncreaseType::Multiplicative(1.03)
                } else {
                    let rtt_ms = rtt * 1000.;
                    let response_time_ms = 100. + rtt_ms;
                    let time_since_last_update_ms = match last_update_time {
                        None => 0.,
                        Some(instant) => {
                            (self.last_update_time.unwrap() - instant).as_millis() as f64
                        }
                    };
                    // gcc section 5.5 advises 0.95 as the smoothing factor, but that
                    // seems intuitively much too low, granting disproportionate importance
                    // to the last measurement. 0.5 seems plenty enough, I don't have maths
                    // to back that up though :)
                    // TODO: @viotovmp - potentially reset to 0.95 from 0.5
                    let alpha = 0.5 * f64::min(time_since_last_update_ms / response_time_ms, 1.0);
                    // TODO: @viotovmp - 10 should come from the framerate of the stream
                    let bits_per_frame = target_bitrate / 10.;
                    // TODO: @viotovmp - 1200 should come from the webrtc mtu
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
                IncreaseType::Multiplicative(1.03)
            })
        }
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
        let loss_percentage = stats.get::<f64>("packet-loss-pct").unwrap();

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
            let op = self.update_delay(element, &twcc_stats, self.lookup_rtt(stats));
            self.apply_control_op(element, encoders, op, ControllerType::Delay);
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

            if self.do_apply {
                let fec_ratio = {
                    if self.do_fec {
                        if target_bitrate <= 2000000 || self.max_bitrate <= 2000000 {
                            0f64
                        } else {
                            (target_bitrate as f64 - 2000000f64)
                                / (self.max_bitrate as f64 - 2000000f64)
                        }
                    } else {
                        0.0
                    }
                };

                let fec_percentage = (fec_ratio * 50f64) as u32;

                for encoder in encoders.iter_mut() {
                    if encoder.set_bitrate(element, target_bitrate).is_ok() {
                        encoder
                            .transceiver
                            .set_property("fec-percentage", fec_percentage);
                    }
                }
            }

            // Emit a signal through the parent `webrtcsink`
            element.emit_by_name::<()>(
                "cc-bitrate-changed",
                &[
                    &(self.target_bitrate_on_delay as u32),
                    &(self.target_bitrate_on_loss as u32),
                    &(target_bitrate as u32),
                ],
            );
        }
    }
}
