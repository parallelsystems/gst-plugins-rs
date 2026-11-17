// SPDX-License-Identifier: MPL-2.0

//! The homegrown congestion controller (`WebRTCSinkCongestionControl::Homegrown`):
//! a delay controller and a loss controller on the aggregate TWCC stats, the
//! lower of their targets applied to the encoders.

use gst::{
    glib::{self, value::FromValue},
    prelude::*,
};
use once_cell::sync::Lazy;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

use super::imp::VideoEncoder;

static CAT: Lazy<gst::DebugCategory> = Lazy::new(|| {
    gst::DebugCategory::new(
        "webrtcsink-homegrowncc",
        gst::DebugColorFlags::empty(),
        Some("WebRTC sink"),
    )
});

/// The delay signals are averaged over a sliding window one GOP long, read
/// from the encoders' keyframe interval and current framerate, before being
/// compared against their thresholds.
///
/// The TWCC stats are computed over a short window, and the sender doesn't
/// pace, so every packet of a frame gets roughly the same send timestamp
/// while the receiver sees them spread out by the time it takes the frame to
/// drain through the bottleneck. Whenever the newest frame in that window is
/// a keyframe, bitrate-recv reads noticeably lower than bitrate-sent even on
/// an idle link: without averaging, a spurious decrease once per keyframe.
///
/// No decision is taken until a full window has been collected after each
/// decrease, so the window is bounded to keep the controller responsive with
/// long or unbounded GOPs.
const DEFAULT_DELAY_SIGNAL_WINDOW: Duration = Duration::from_millis(1000);
const MIN_DELAY_SIGNAL_WINDOW: Duration = Duration::from_millis(500);
const MAX_DELAY_SIGNAL_WINDOW: Duration = Duration::from_secs(3);

/// Delay signal window covering the longest GOP among the encoders, or the
/// default when no encoder has a known GOP
fn delay_signal_window(encoders: &[VideoEncoder]) -> Duration {
    encoders
        .iter()
        .filter_map(VideoEncoder::gop_duration)
        .max()
        .map(|gop| gop.clamp(MIN_DELAY_SIGNAL_WINDOW, MAX_DELAY_SIGNAL_WINDOW))
        .unwrap_or(DEFAULT_DELAY_SIGNAL_WINDOW)
}

/// Averaged (sent - recv) / sent above which the delay controller decreases
const DELAY_FACTOR_THRESHOLD: f64 = 0.1;

/// Averaged avg-delta-of-delta (ns) above which the delay controller decreases
const DELTA_OF_DELTA_THRESHOLD_NS: f64 = 1_000_000.;

/// Factor the delay controller cuts by. With the signals averaged, it
/// decides at most once per window, so each cut has to count.
const DELAY_DECREASE_FACTOR: f64 = 0.85;

/// Deepest single cut, when the averaged delay factor says even less than
/// DELAY_DECREASE_FACTOR of what was sent is getting through
const MIN_DELAY_DECREASE_FACTOR: f64 = 0.5;

/// How far above the applied target either controller's target may climb.
/// The applied target is the lower of the two, so the other one keeps
/// increasing: left alone it runs away to max-bitrate, and later cuts have
/// to work their way down from there before they affect the applied target.
const TARGET_HEADROOM: f64 = 1.1;

/// Multiplicative increase per second, for both controllers (libwebrtc's
/// AIMD rate control uses ~8%/s). Applied scaled to the time since the last
/// update, so it doesn't depend on how often the stats come.
const MULTIPLICATIVE_INCREASE_PER_SEC: f64 = 1.08;

/// Lower bound of the additive increase rate, in bits per second
const MIN_ADDITIVE_INCREASE_BPS_PER_SEC: f64 = 10_000.;

/// Network response time on top of the RTT for the additive increase, as in
/// gcc section 5.5
const RESPONSE_TIME_BASE: Duration = Duration::from_millis(100);

/// Longest interval a single increase is scaled over, so a stall in the
/// stats doesn't turn into one large step
const MAX_INCREASE_INTERVAL: Duration = Duration::from_secs(1);

/// Framerate assumed for the expected packet size when an encoder's
/// framerate is unknown (variable framerate, or not negotiated yet)
const FALLBACK_FRAMERATE: f64 = 30.;

/// Half-width of the congestion window around the bitrate EMA, in standard
/// deviations (gcc section 5.5). Below it the delay controller increases
/// multiplicatively, inside it additively, above it the window is
/// discarded. main used 7: after a few deep cuts the standard deviation
/// covered the whole range, and the controller stayed additive, climbing
/// ~70 kbps/s from the floor (hitl1116 regression).
const CONGESTION_WINDOW_STDDEVS: f64 = 3.;

/// Lower bound on the standard deviation used for the congestion window, as
/// a fraction of the EMA, so a window seeded by one or two decreases isn't
/// nearly zero wide
const MIN_BITRATE_STDDEV_RATIO: f64 = 0.05;

/// Time without a delay decrease after which the congestion window is
/// discarded and the delay controller increases multiplicatively again
const CONGESTION_WINDOW_EXPIRY: Duration = Duration::from_secs(5);

/// Where a target sits against the congestion window
#[derive(Debug, Clone, Copy, PartialEq)]
enum WindowPosition {
    Below,
    Inside,
    Above,
}

/// Where `target` sits against the congestion window around `ema`
fn window_position(target: f64, ema: f64, emvar: f64) -> WindowPosition {
    let stdev = f64::max(emvar.sqrt(), MIN_BITRATE_STDDEV_RATIO * ema);
    if target < ema - CONGESTION_WINDOW_STDDEVS * stdev {
        WindowPosition::Below
    } else if target > ema + CONGESTION_WINDOW_STDDEVS * stdev {
        WindowPosition::Above
    } else {
        WindowPosition::Inside
    }
}

/// How long the delay controller ignores the stats after a decrease before
/// it starts collecting a new window: what was sent at the old bitrate is
/// still queued and in flight for a response time (RESPONSE_TIME_BASE +
/// RTT), and the TWCC stats keep counting it for about a window. Without it
/// the window right after a cut still saw the backlog and cut again, from
/// 3.7 to 0.6 Mbps in 2 s on a step to 3 Mbps (hitl1116 regression).
fn delay_settle_time(rtt: f64, window: Duration) -> Duration {
    RESPONSE_TIME_BASE + Duration::from_secs_f64(rtt.max(0.)) + window
}

/// Stats the loss controller waits for after a cut, on top of the response
/// time, before it judges the loss again: TWCC's packet-loss-pct keeps
/// counting what was lost at the old bitrate for about this long
const LOSS_STATS_WINDOW: Duration = Duration::from_secs(1);

/// How long the loss controller holds after a decrease. The loss reports in
/// the meantime are about packets sent at the old bitrate, still queued or
/// in flight, and cutting on each of them cuts again for the same loss:
/// seven 30% cuts in 1.5 s took a step from 6 to 3 Mbps down to the 500 kbps
/// floor (hitl1116 regression rev1).
fn loss_settle_time(rtt: f64) -> Duration {
    RESPONSE_TIME_BASE + Duration::from_secs_f64(rtt.max(0.)) + LOSS_STATS_WINDOW
}

/// Seconds since `last`, bounded by MAX_INCREASE_INTERVAL, 0 on the first update
fn increase_interval_secs(last: Option<Instant>, now: Instant) -> f64 {
    last.map_or(0., |last| {
        now.duration_since(last)
            .min(MAX_INCREASE_INTERVAL)
            .as_secs_f64()
    })
}

/// Multiplicative increase at MULTIPLICATIVE_INCREASE_PER_SEC over `interval_secs`
fn multiplicative_increase(interval_secs: f64) -> IncreaseType {
    IncreaseType::Multiplicative(MULTIPLICATIVE_INCREASE_PER_SEC.powf(interval_secs))
}

/// Sliding-window average of the delay controller's input signals
#[derive(Debug)]
struct DelaySignals {
    /// (sample time, signed (sent - recv) / sent, avg-delta-of-delta in ns)
    samples: VecDeque<(Instant, f64, f64)>,
    /// When collection started since the last reset
    since: Option<Instant>,
    /// Length of the averaging window
    window: Duration,
}

impl Default for DelaySignals {
    fn default() -> Self {
        Self {
            samples: VecDeque::new(),
            since: None,
            window: DEFAULT_DELAY_SIGNAL_WINDOW,
        }
    }
}

impl DelaySignals {
    /// Returns whether the window changed. A shorter window drops the
    /// excess samples on the next push, a longer one delays the next
    /// decision until it has been filled since the last reset.
    fn set_window(&mut self, window: Duration) -> bool {
        std::mem::replace(&mut self.window, window) != window
    }

    fn push(&mut self, now: Instant, rel_diff: f64, delta_of_delta: f64) {
        self.since.get_or_insert(now);
        self.samples.push_back((now, rel_diff, delta_of_delta));
        while let Some(&(t, ..)) = self.samples.front() {
            if now.duration_since(t) >= self.window {
                self.samples.pop_front();
            } else {
                break;
            }
        }
    }

    /// Mean (delay_factor, delta_of_delta) over the window, or None until
    /// samples have been collected for a full window since the last reset
    fn averages(&self, now: Instant) -> Option<(f64, f64)> {
        let since = self.since?;
        if now.duration_since(since) < self.window || self.samples.is_empty() {
            return None;
        }

        let n = self.samples.len() as f64;
        let (rel_sum, dod_sum) = self
            .samples
            .iter()
            .fold((0., 0.), |(r, d), &(_, rel, dod)| (r + rel, d + dod));

        Some((rel_sum / n, dod_sum / n))
    }

    /// Drops all samples, after a decrease: samples taken at the old bitrate
    /// say nothing about the new one, and keeping them would make the
    /// controller cut again on every update until they age out
    fn reset(&mut self) {
        self.samples.clear();
        self.since = None;
    }
}

/// Cut for an averaged delay factor above DELAY_FACTOR_THRESHOLD: the
/// factor is the share of what was sent that isn't getting through, so cut
/// at least down to what is, but by no more than half
fn delay_decrease_factor(delay_factor: f64) -> f64 {
    f64::min(DELAY_DECREASE_FACTOR, 1. - delay_factor).max(MIN_DELAY_DECREASE_FACTOR)
}

#[derive(Debug, Clone, Copy, PartialEq)]
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

/// The new target of the controller whose target is `current`, after `op`.
/// Increases stay within TARGET_HEADROOM of the applied target
/// `prev_applied`, which can also pull a target down when the other
/// controller has lowered the applied target since. Not clamped to the
/// min/max bitrate yet.
fn next_target(current: i32, op: &CongestionControlOp, prev_applied: i32) -> i32 {
    let cap_increase =
        |bitrate: i32| i32::min(bitrate, (prev_applied as f64 * TARGET_HEADROOM) as i32);
    match op {
        CongestionControlOp::Hold => current,
        CongestionControlOp::Increase(IncreaseType::Additive(value)) => {
            cap_increase(current + *value as i32)
        }
        CongestionControlOp::Increase(IncreaseType::Multiplicative(factor)) => {
            cap_increase((current as f64 * factor) as i32)
        }
        CongestionControlOp::Decrease { factor, .. } => (current as f64 * factor) as i32,
    }
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
    /// Time of the last delay controller update, increases are scaled to
    /// the time elapsed since
    last_update_time: Option<Instant>,
    /// Time of the last loss controller update, same purpose
    last_loss_update_time: Option<Instant>,
    /// Time of the last delay decrease: the delay controller settles after
    /// it, and the congestion window expires without another
    last_delay_decrease: Option<Instant>,
    /// Time of the last loss decrease: the loss controller settles after it
    last_loss_decrease: Option<Instant>,
    /// Latest round-trip time from the stats, seconds
    rtt: f64,
    /// Delay signals averaged over a GOP-long window before thresholding
    delay_signals: DelaySignals,
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
            last_loss_update_time: None,
            last_delay_decrease: None,
            last_loss_decrease: None,
            rtt: 0.,
            delay_signals: DelaySignals::default(),
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
        encoders: &[VideoEncoder],
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
        let interval_secs = increase_interval_secs(self.last_update_time.replace(now), now);

        if self.delay_signals.set_window(delay_signal_window(encoders)) {
            gst::debug!(
                CAT,
                obj = element,
                "consumer {}: delay signal window now {:?}",
                self.peer_id,
                self.delay_signals.window
            );
        }

        gst::trace!(
            CAT,
            obj = element,
            "consumer {}: considering stats {}",
            self.peer_id,
            twcc_stats
        );

        let settle = delay_settle_time(rtt, self.delay_signals.window);
        if self
            .last_delay_decrease
            .is_some_and(|t| now.duration_since(t) < settle)
        {
            gst::trace!(
                CAT,
                obj = element,
                "consumer {}: settling for {:?} after the last delay decrease, holding",
                self.peer_id,
                settle
            );
            return CongestionControlOp::Hold;
        }

        // Signed on purpose: burst-drain artifacts push recv both above and
        // below sent, and only a sustained excess of sent over recv means
        // congestion. Clamping each sample to >= 0 before averaging would
        // turn that zero-mean noise into a positive bias.
        if bitrate_sent > 0 {
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
            self.delay_signals.reset();
            CongestionControlOp::Decrease {
                factor: delay_decrease_factor(delay_factor),
                reason: format!("delay factor {delay_factor}"),
            }
        } else if delta_of_delta > DELTA_OF_DELTA_THRESHOLD_NS {
            self.delay_signals.reset();
            CongestionControlOp::Decrease {
                factor: DELAY_DECREASE_FACTOR,
                reason: format!("High delta: {delta_of_delta}"),
            }
        } else {
            if self.bitrate_ema.is_some()
                && self
                    .last_delay_decrease
                    .is_some_and(|t| now.duration_since(t) >= CONGESTION_WINDOW_EXPIRY)
            {
                gst::debug!(
                    CAT,
                    obj = element,
                    "consumer {}: no delay decrease for {:?}, discarding congestion window",
                    self.peer_id,
                    CONGESTION_WINDOW_EXPIRY
                );
                self.bitrate_ema = None;
            }

            CongestionControlOp::Increase(if let Some(ema) = self.bitrate_ema {
                gst::trace!(
                    CAT,
                    obj = element,
                    "consumer {}: Old bitrate: {}, ema: {}, stddev: {}",
                    self.peer_id,
                    target_bitrate,
                    ema,
                    self.bitrate_emvar.sqrt(),
                );

                match window_position(target_bitrate, ema, self.bitrate_emvar) {
                    WindowPosition::Below => {
                        gst::trace!(
                            CAT,
                            obj = element,
                            "consumer {}: below last congestion window",
                            self.peer_id
                        );
                        /* Multiplicative increase */
                        multiplicative_increase(interval_secs)
                    }
                    WindowPosition::Above => {
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
                        multiplicative_increase(interval_secs)
                    }
                    WindowPosition::Inside => {
                        gst::trace!(
                            CAT,
                            obj = element,
                            "consumer {}: still in last congestion window",
                            self.peer_id,
                        );

                        /* Additive increase */
                        self.additive_increase(
                            element,
                            target_bitrate,
                            interval_secs,
                            rtt,
                            encoders,
                        )
                    }
                }
            } else {
                /* Multiplicative increase */
                gst::trace!(
                    CAT,
                    obj = element,
                    "consumer {}: outside congestion window",
                    self.peer_id
                );
                multiplicative_increase(interval_secs)
            })
        }
    }

    /// About one packet per response time, as in gcc section 5.5 and
    /// libwebrtc, scaled to the time since the last update
    fn additive_increase(
        &self,
        element: &super::BaseWebRTCSink,
        target_bitrate: f64,
        interval_secs: f64,
        rtt: f64,
        encoders: &[VideoEncoder],
    ) -> IncreaseType {
        let response_time_secs = RESPONSE_TIME_BASE.as_secs_f64() + rtt;
        let avg_packet_size_bits = self.avg_packet_size_bits(element, target_bitrate, encoders);
        let rate_bps_per_sec = f64::max(
            MIN_ADDITIVE_INCREASE_BPS_PER_SEC,
            avg_packet_size_bits / response_time_secs,
        );

        gst::trace!(
            CAT,
            obj = element,
            "consumer {}: increasing additively by {:.0} bps/s",
            self.peer_id,
            rate_bps_per_sec,
        );

        IncreaseType::Additive(rate_bps_per_sec * interval_secs)
    }

    /// Expected average RTP packet size in bits, averaged over the encoders,
    /// from the framerate each encoder is fed and its payloader's MTU.
    /// `target_bitrate` is the total for all encoders and is split equally,
    /// as in apply_control_op.
    fn avg_packet_size_bits(
        &self,
        element: &super::BaseWebRTCSink,
        target_bitrate: f64,
        encoders: &[VideoEncoder],
    ) -> f64 {
        if encoders.is_empty() {
            return 0.;
        }

        let bitrate_per_encoder = target_bitrate / encoders.len() as f64;
        let sum: f64 = encoders
            .iter()
            .map(|encoder| {
                let framerate = encoder.framerate().unwrap_or(FALLBACK_FRAMERATE);
                let packet_size_bits =
                    packet_size_bits(bitrate_per_encoder, framerate, encoder.mtu());

                gst::trace!(
                    CAT,
                    obj = element,
                    "consumer {}: framerate {}, mtu {}, {:.0} bit packets",
                    self.peer_id,
                    framerate,
                    encoder.mtu(),
                    packet_size_bits,
                );

                packet_size_bits
            })
            .sum();

        sum / encoders.len() as f64
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
        let now = Instant::now();
        let interval_secs = increase_interval_secs(self.last_loss_update_time.replace(now), now);

        let settle = loss_settle_time(self.rtt);
        if self
            .last_loss_decrease
            .is_some_and(|t| now.duration_since(t) < settle)
        {
            gst::trace!(
                CAT,
                obj = element,
                "consumer {}: loss {loss_percentage:.1}%, settling for {:?} after the last loss decrease, holding",
                self.peer_id,
                settle
            );
            return;
        }

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
                CongestionControlOp::Increase(multiplicative_increase(interval_secs))
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
            let rtt = self.lookup_rtt(stats);
            self.rtt = rtt;
            let op = self.update_delay(element, &twcc_stats, rtt, encoders);
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

        if encoders.is_empty() {
            return;
        }
        let n_encoders = encoders.len() as i32;
        let prev_bitrate = i32::min(self.target_bitrate_on_delay, self.target_bitrate_on_loss);
        // Each controller moves its own target
        let current_bitrate = match controller_type {
            ControllerType::Delay => self.target_bitrate_on_delay,
            ControllerType::Loss => self.target_bitrate_on_loss,
        };
        self.clamp_bitrate(
            next_target(current_bitrate, &control_op, prev_bitrate),
            n_encoders,
            controller_type,
        );

        if let (CongestionControlOp::Decrease { .. }, ControllerType::Loss) =
            (&control_op, controller_type)
        {
            self.last_loss_decrease = Some(Instant::now());
        }

        if let (CongestionControlOp::Decrease { .. }, ControllerType::Delay) =
            (&control_op, controller_type)
        {
            self.last_delay_decrease = Some(Instant::now());
            // Smoothing factor
            let alpha = 0.75;
            if let Some(ema) = self.bitrate_ema {
                let sigma: f64 = (self.target_bitrate_on_delay as f64) - ema;
                self.bitrate_ema = Some(ema + (alpha * sigma));
                self.bitrate_emvar = (1. - alpha) * (self.bitrate_emvar + alpha * sigma.powi(2));
            } else {
                self.bitrate_ema = Some(self.target_bitrate_on_delay as f64);
                self.bitrate_emvar = 0.;
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

/// Expected RTP packet size in bits for a stream at `bitrate` and
/// `framerate`, packetized at `mtu` bytes: one frame's worth split evenly
/// over the packets it takes
fn packet_size_bits(bitrate: f64, framerate: f64, mtu: u32) -> f64 {
    let bits_per_frame = bitrate / framerate;
    let packets_per_frame = f64::max(1., f64::ceil(bits_per_frame / (f64::from(mtu) * 8.)));
    bits_per_frame / packets_per_frame
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decrease(factor: f64) -> CongestionControlOp {
        CongestionControlOp::Decrease {
            factor,
            reason: String::new(),
        }
    }

    #[test]
    fn each_controller_moves_its_own_target() {
        // main applied every operation to the delay target: with delay at
        // 3 Mbps and loss at 1 Mbps, a loss increase gave the loss
        // controller 1.05 x 3 Mbps
        let inc = CongestionControlOp::Increase(IncreaseType::Multiplicative(1.05));
        assert_eq!(next_target(1_000_000, &inc, 1_000_000), 1_050_000);
        assert_eq!(next_target(1_000_000, &decrease(0.85), 1_000_000), 850_000);
        assert_eq!(
            next_target(1_000_000, &CongestionControlOp::Hold, 1_000_000),
            1_000_000
        );
    }

    #[test]
    fn increases_stay_within_headroom_of_the_applied_target() {
        // the delay target can't run away while loss holds the applied one
        let inc = CongestionControlOp::Increase(IncreaseType::Multiplicative(1.08));
        assert_eq!(next_target(1_500_000, &inc, 1_400_000), 1_540_000);
        let inc = CongestionControlOp::Increase(IncreaseType::Additive(10_000.));
        assert_eq!(next_target(1_000_000, &inc, 1_000_000), 1_010_000);
        // and gets pulled back down when the applied target fell since
        assert_eq!(next_target(3_000_000, &inc, 1_000_000), 1_100_000);
        // decreases aren't capped
        assert_eq!(next_target(3_000_000, &decrease(0.5), 1_000_000), 1_500_000);
    }

    #[test]
    fn increases_scale_with_time() {
        let t0 = Instant::now();
        assert_eq!(increase_interval_secs(None, t0), 0.);
        let half = increase_interval_secs(Some(t0), t0 + Duration::from_millis(500));
        assert!((half - 0.5).abs() < 1e-9);
        // a stall in the stats is one step of at most a second
        assert_eq!(
            increase_interval_secs(Some(t0), t0 + Duration::from_secs(10)),
            1.
        );
        let IncreaseType::Multiplicative(f) = multiplicative_increase(1.) else {
            panic!()
        };
        assert!((f - MULTIPLICATIVE_INCREASE_PER_SEC).abs() < 1e-9);
        // two half-second steps make one second's increase
        let IncreaseType::Multiplicative(f) = multiplicative_increase(0.5) else {
            panic!()
        };
        assert!((f * f - MULTIPLICATIVE_INCREASE_PER_SEC).abs() < 1e-9);
    }

    #[test]
    fn packet_size_follows_framerate_and_mtu() {
        // 2 Mbps at 10 fps: 200 kbit frames in 21 packets of 1200 bytes
        let bits = packet_size_bits(2e6, 10., 1200);
        assert!((bits - 200_000. / 21.).abs() < 1e-6);
        // at the 30 fps main assumed: 66.7 kbit frames, 7 packets
        let bits = packet_size_bits(2e6, 30., 1200);
        assert!((bits - 2e6 / 30. / 7.).abs() < 1e-6);
        // a frame smaller than a packet is one packet
        assert!((packet_size_bits(100e3, 30., 1200) - 100e3 / 30.).abs() < 1e-6);
    }

    #[test]
    fn delay_signals_average_over_the_window() {
        let t0 = Instant::now();
        let ms = Duration::from_millis;
        let mut signals = DelaySignals::default();
        assert!(!signals.set_window(DEFAULT_DELAY_SIGNAL_WINDOW));
        // a keyframe tick reads high, the rest of the window negative
        signals.push(t0, 0.4, 0.);
        assert_eq!(signals.averages(t0 + ms(500)), None);
        for i in 1..10 {
            signals.push(t0 + ms(100 * i), -0.04, 0.);
        }
        let (factor, _) = signals.averages(t0 + ms(1000)).unwrap();
        assert!((factor - (0.4 - 9. * 0.04) / 10.).abs() < 1e-9);
        assert!(factor < DELAY_FACTOR_THRESHOLD);
        // after a cut, nothing until a full window again
        signals.reset();
        signals.push(t0 + ms(1100), 0.5, 0.);
        assert_eq!(signals.averages(t0 + ms(1500)), None);
    }

    #[test]
    fn congestion_window_is_three_stddevs_wide() {
        // the regression run: an EMA of ~2 Mbps built from cuts of a few
        // Mbps each. At 7 sigma the window covered the floor to the max;
        // at 3 the climb from the floor is multiplicative.
        let ema = 2e6;
        let emvar = 0.6e6_f64.powi(2);
        assert_eq!(window_position(0.5e6, ema, emvar), WindowPosition::Inside);
        let emvar = 0.4e6_f64.powi(2);
        assert_eq!(window_position(0.5e6, ema, emvar), WindowPosition::Below);
        assert_eq!(window_position(2.5e6, ema, emvar), WindowPosition::Inside);
        assert_eq!(window_position(3.5e6, ema, emvar), WindowPosition::Above);
        // a window seeded by one cut (no variance) is still 5% wide
        assert_eq!(window_position(2.2e6, ema, 0.), WindowPosition::Inside);
        assert_eq!(window_position(2.4e6, ema, 0.), WindowPosition::Above);
        assert_eq!(window_position(1.6e6, ema, 0.), WindowPosition::Below);
    }

    #[test]
    fn delay_controller_settles_for_a_response_time_and_a_window() {
        let window = Duration::from_millis(500);
        assert_eq!(delay_settle_time(0., window), Duration::from_millis(600));
        assert_eq!(delay_settle_time(0.25, window), Duration::from_millis(850));
        // with the 2.5 s window of a 25 frame GOP at 10 fps
        assert_eq!(
            delay_settle_time(0.125, Duration::from_millis(2500)),
            Duration::from_millis(2725)
        );
        assert_eq!(delay_settle_time(-1., window), Duration::from_millis(600));
    }

    #[test]
    fn loss_controller_settles_for_a_response_time_and_a_second() {
        // no RTT from the stats (Janus sends no receiver reports): 1.1 s
        assert_eq!(loss_settle_time(0.), Duration::from_millis(1100));
        assert_eq!(loss_settle_time(0.25), Duration::from_millis(1350));
        assert_eq!(loss_settle_time(-1.), Duration::from_millis(1100));
        // the 6 to 3 Mbps step: one 0.70 cut per settle period instead of
        // one per stats update, 4.5 Mbps reaches ~1.1 Mbps in four periods
        // rather than the floor in 1.5 s
        let mut bitrate = 4_500_000;
        let cut = CongestionControlOp::Decrease {
            factor: 0.7,
            reason: String::new(),
        };
        for _ in 0..4 {
            bitrate = next_target(bitrate, &cut, bitrate);
        }
        assert!((1_000_000..1_200_000).contains(&bitrate));
    }

    #[test]
    fn delay_cuts_follow_what_gets_through() {
        assert_eq!(delay_decrease_factor(0.12), DELAY_DECREASE_FACTOR);
        assert!((delay_decrease_factor(0.3) - 0.7).abs() < 1e-9);
        assert_eq!(delay_decrease_factor(0.8), MIN_DELAY_DECREASE_FACTOR);
    }
}
