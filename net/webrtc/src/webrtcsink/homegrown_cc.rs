// SPDX-License-Identifier: MPL-2.0

use gst::{
    glib::{self, value::FromValue},
    prelude::*,
};
use once_cell::sync::Lazy;

use super::imp::VideoEncoder;
use super::imp::LOW_FRAMERATE_THRESHOLD_BITRATE;

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

    /// The actual target bitrate, the min of the above two,
    /// clamped between `min_bitrate` and `max_bitrate`.
    pub target_bitrate: i32,

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
    /// For logging purposes
    peer_id: String,

    min_bitrate: u32,
    max_bitrate: u32,
    do_fec: bool,

    /// List of packet loss reported by the loss controller;
    /// drained whenever stats are generated
    losses: Vec<f64>
}

/// Exponential Moving Average weights
/// 0.95 is recommended; [0,1] higher = more weight on recent measurements
const SMOOTHING_FACTOR: f64 = 0.85;
/// Inter-packet spacing at 20FPS (50 msec on avg)
const PKT_DELTA_OF_DELTAS_FULL_FPS_NSEC: i64 = 1_000_000;
/// Inter-packet spacing at 10FPS (100 msec on avg)
const PKT_DELTA_OF_DELTAS_LOW_FPS_NSEC: i64 = 2_000_000;
/// The number of std-deviations to use as a range for the EMA
const NUM_STD_DEV_EMA: f64 = 3.0;

impl CongestionController {
    pub fn new(peer_id: &str, min_bitrate: u32, max_bitrate: u32, do_fec: bool) -> Self {
        Self {
            target_bitrate_on_delay: 0,
            target_bitrate_on_loss: 0,
            target_bitrate: 0,
            bitrate_ema: None,
            bitrate_emvar: 0.,
            last_update_time: None,
            peer_id: peer_id.to_string(),
            min_bitrate,
            max_bitrate,
            do_fec,
            losses: vec![]
        }
    }

    fn update_delay(
        &mut self,
        element: &super::BaseWebRTCSink,
        twcc_stats: &gst::StructureRef,
        rtt: f64,
    ) -> CongestionControlOp {
        let target_bitrate = self.target_bitrate as f64;

        // Unwrap, all those fields must be there or there's been an API
        // break, which qualifies as programming error
        let bitrate_sent = twcc_stats.get::<u32>("bitrate-sent").unwrap();
        let bitrate_recv = twcc_stats.get::<u32>("bitrate-recv").unwrap();
        let delta_of_delta = twcc_stats.get::<i64>("avg-delta-of-delta").unwrap();

        let packets_sent = twcc_stats.get::<u32>("packets-sent").unwrap_or(0);
        let packets_recv = twcc_stats.get::<u32>("packets-recv").unwrap_or(0);

        let sent_minus_received = bitrate_sent.saturating_sub(bitrate_recv);

        gst::info!(CAT, obj = element,
            "DELAY controller: {rtt} sec ----  bitrate sent {bitrate_sent} recv {bitrate_recv} ==> delta {sent_minus_received} ---- target bitrate {target_bitrate} (L: {}, D:{}) ",
            self.target_bitrate_on_loss, self.target_bitrate_on_delay
        );
        gst::info!(CAT, obj = element,
            "               packets sent {packets_sent} recv {packets_recv} ==> delta of deltas {delta_of_delta}",
        );

        // How much of the target bitrate did we lose?
        let delay_factor = sent_minus_received as f64 / target_bitrate;
        let last_update_time = self.last_update_time.replace(std::time::Instant::now());

        let threshold_delta_of_deltas = if target_bitrate < (LOW_FRAMERATE_THRESHOLD_BITRATE as f64) {
            PKT_DELTA_OF_DELTAS_LOW_FPS_NSEC
        } else {
            PKT_DELTA_OF_DELTAS_FULL_FPS_NSEC
        };

        // If we've lost >10%
        if delay_factor > 0.1 {
            let (factor, reason) = if delay_factor < 0.64 {
                // 10-64% -> low loss
                (0.96, format!("low delay factor {delay_factor}"))
            } else {
                // More than 64% -> high loss
                (
                    delay_factor.sqrt().sqrt().clamp(0.8, 0.96),
                    format!("High delay factor {delay_factor}"),
                )
            };

            gst::warning!(CAT, "Delay factor {factor} -- decreasing");

            CongestionControlOp::Decrease { factor, reason }
        } else if delta_of_delta > threshold_delta_of_deltas {
            gst::warning!(CAT, "Interpacket delta-of-deltas exceeded threshold {threshold_delta_of_deltas} -> {delta_of_delta} ns");
            CongestionControlOp::Decrease {
                factor: 0.97,
                reason: format!("High delta: {delta_of_delta}"),
            }
        } else {
            // Exponential moving average
            let t = if let Some(ema) = self.bitrate_ema {
                let bitrate_stdev = self.bitrate_emvar.sqrt();

                gst::info!(
                    CAT,
                    "[Checking envelope] Old bitrate: {}, ema: {}, stddev: {} --> [{} / {}]",
                    target_bitrate,
                    ema,
                    bitrate_stdev,
                    ema - NUM_STD_DEV_EMA * bitrate_stdev,
                    ema + NUM_STD_DEV_EMA * bitrate_stdev,
                );

                if target_bitrate < ema - NUM_STD_DEV_EMA * bitrate_stdev {
                    gst::info!(CAT, obj = element, "BELOW last congestion window");
                    /* Multiplicative increase */
                    IncreaseType::Multiplicative(1.03)
                } else if target_bitrate > ema + NUM_STD_DEV_EMA * bitrate_stdev {
                    gst::warning!(CAT, obj = element, "ABOVE last congestion window -- resetting MOVING AVERAGE!");
                    /* We have gone past our last estimated max bandwidth
                     * network situation may have changed, go back to
                     * multiplicative increase
                     */
                    self.bitrate_ema.take();
                    IncreaseType::Multiplicative(1.03)
                } else {
                    // We're within the bounds of the expected congestion window...
                    let rtt_ms = rtt * 1000.;
                    let response_time_ms = 100. + rtt_ms; // Google says to add 100msec
                    let time_since_last_update_ms = match last_update_time {
                        None => 0.,
                        Some(instant) => {
                            (self.last_update_time.unwrap() - instant).as_millis() as f64
                        }
                    };

                    // gcc section 5.5 advises 0.95 as the smoothing factor
                    let alpha = SMOOTHING_FACTOR * f64::min(time_since_last_update_ms / response_time_ms, 1.0);

                    // Calculate how big each encoded frame is according to the framerate
                    let stream_framerate = if target_bitrate < (LOW_FRAMERATE_THRESHOLD_BITRATE as f64) {
                        10.
                    } else {
                        20.
                    };

                    let bits_per_frame = target_bitrate / stream_framerate;
                    // 1200 = avg RTP packet size
                    // 8 converts from bits to bytes
                    let avg_rtp_packet_bits = 1200. * 8.;
                    let packets_per_frame = f64::ceil(bits_per_frame / avg_rtp_packet_bits);
                    let avg_packet_size_bits = bits_per_frame / packets_per_frame;

                    // Cautiously grow (additive increase) within the congestion window
                    let added_bits = f64::max(1000., alpha * avg_packet_size_bits);
                    gst::info!(CAT, obj = element, "Within delay congestion window (@ {stream_framerate} fps) - adding {added_bits} bits to allowance");
                    IncreaseType::Additive(added_bits)
                }
            } else {
                /* Multiplicative increase */
                gst::warning!(CAT, obj = element, "outside congestion window -- increasing");
                IncreaseType::Multiplicative(1.03)
            };

            // How large of a window is this?
            CongestionControlOp::Increase(t)
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

        rtt
    }

    pub fn loss_control(
        &mut self,
        element: &super::BaseWebRTCSink,
        _session_stats: &gst::StructureRef,
        twcc_stats: &gst::StructureRef,
        encoders: &mut [VideoEncoder],
    ) {
        let twcc_loss_percentage = twcc_stats.get::<f64>("packet-loss-pct").unwrap();
        self.losses.push(twcc_loss_percentage);

        gst::warning!(
            CAT,
            "LOSS controller - {twcc_loss_percentage}% packet loss - (Target Loss BR {} Delay BR {})",
            self.target_bitrate_on_loss,
            self.target_bitrate_on_delay
        );

        self.apply_control_op(
            element,
            encoders,
            if twcc_loss_percentage > 10. {
                CongestionControlOp::Decrease {
                    factor: ((100. - (0.5 * twcc_loss_percentage)) / 100.).clamp(0.7, 0.98),
                    reason: format!("High loss: {twcc_loss_percentage}"),
                }
            } else if twcc_loss_percentage > 2. {
                CongestionControlOp::Hold
            } else {
                CongestionControlOp::Increase(IncreaseType::Multiplicative(1.05))
            },
            ControllerType::Loss,
        );
    }

    /// Runs every 100 msec as a `glib::promise` to `get-stats` on `webrtcbin`
    pub fn delay_control(
        &mut self,
        element: &super::BaseWebRTCSink,
        stats: &gst::StructureRef,
        encoders: &mut [VideoEncoder],
    ) {
        if let Some(twcc_stats) = lookup_twcc_stats(stats) {
            let rtt = self.lookup_rtt(stats);
            let op = self.update_delay(element, &twcc_stats, rtt);
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
        let n_encoders = encoders.len() as i32;
        let prev_bitrate = match controller_type {
            ControllerType::Delay => self.target_bitrate_on_delay,
            ControllerType::Loss => self.target_bitrate_on_loss,
        };
        match &control_op {
            CongestionControlOp::Hold => { }
            CongestionControlOp::Increase(IncreaseType::Additive(value)) => {
                self.clamp_bitrate(
                    // self.target_bitrate_on_delay + *value as i32,
                    prev_bitrate + *value as i32,
                    n_encoders,
                    controller_type,
                );
            }
            CongestionControlOp::Increase(IncreaseType::Multiplicative(factor)) => {
                self.clamp_bitrate(
                    // (self.target_bitrate_on_delay as f64 * factor) as i32,
                    (prev_bitrate as f64 * factor) as i32,
                    n_encoders,
                    controller_type,
                );
            }
            CongestionControlOp::Decrease { factor, .. } => {
                self.clamp_bitrate(
                    // (self.target_bitrate_on_delay as f64 * factor) as i32,
                    (prev_bitrate as f64 * factor) as i32,
                    n_encoders,
                    controller_type,
                );

                // Handle EMA here...
                if let ControllerType::Delay = controller_type {
                    // Smoothing factor - higher gives more weight to recent measurements
                    // let alpha = 0.85;
                    if let Some(ema) = self.bitrate_ema {
                        let sigma: f64 = (self.target_bitrate_on_delay as f64) - ema;
                        self.bitrate_ema = Some(ema + (SMOOTHING_FACTOR * sigma));
                        self.bitrate_emvar =
                            (1. - SMOOTHING_FACTOR) * (self.bitrate_emvar + SMOOTHING_FACTOR * sigma.powi(2));

                            gst::warning!(CAT, "EMA {:?}   VAR {:?}", self.bitrate_ema, self.bitrate_emvar);                        
                    } else {
                        self.bitrate_ema = Some(self.target_bitrate_on_delay as f64);
                        self.bitrate_emvar = 0.;
                    }
                }
            }
        }

        self.target_bitrate =
            i32::min(self.target_bitrate_on_delay, self.target_bitrate_on_loss).clamp(
                self.min_bitrate as i32 * n_encoders,
                self.max_bitrate as i32 * n_encoders,
            ) / n_encoders;

        if self.target_bitrate != prev_bitrate {
            gst::warning!(
                CAT,
                "{prev_bitrate} => {} | delay {} - loss {}",
                self.target_bitrate,
                self.target_bitrate_on_delay,
                self.target_bitrate_on_loss,
            );
        } else {
            gst::warning!(CAT, "Holding same bitrate => {} | delay {} - loss {}",
                self.target_bitrate,
                self.target_bitrate_on_delay,
                self.target_bitrate_on_loss
            );
        }

        let fec_ratio = {
            if self.do_fec {
                if self.target_bitrate <= 2000000 || self.max_bitrate <= 2000000 {
                    0f64
                } else {
                    (self.target_bitrate as f64 - 2000000f64) / (self.max_bitrate as f64 - 2000000f64)
                }
            } else {
                0.0
            }   
        };

        let fec_percentage = (fec_ratio * 50f64) as u32;

        for encoder in encoders.iter_mut() {
            if encoder.set_bitrate(element, self.target_bitrate).is_ok() {
                gst::warning!(CAT, ">>>>>> SET TARGET BITRATE FOR ENCODER {} -- {} Mbps", self.target_bitrate, self.target_bitrate as f64/1_000_000.0);
                encoder
                    .transceiver
                    .set_property("fec-percentage", fec_percentage);
            }
        }
    }

    pub fn average_losses(&mut self) -> f64 {
        let len = self.losses.len();
        if len > 0 {
            let aggregate: f64 = self.losses.drain(..).sum();
            aggregate / (len as f64)
        } else {
            0.
        }
    }
}
