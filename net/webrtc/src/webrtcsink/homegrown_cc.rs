// SPDX-License-Identifier: MPL-2.0

use gst::{
    glib::{self, value::FromValue},
    prelude::*,
};
use once_cell::sync::Lazy;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

use super::imp::VideoEncoder;
use super::packet_feedback::{self, PacketDetector, PacketKind, Signals};

/// The sliding window over which the delay signals are averaged before being
/// compared against their thresholds is one GOP long, read from the encoders'
/// keyframe interval and current framerate.
///
/// The TWCC stats are computed over a short window, and the sender doesn't pace,
/// so every packet of a frame gets roughly the same send timestamp while the
/// receiver sees them spread out by the time it takes the frame to drain
/// through the bottleneck. Whenever the newest frame in that window is a keyframe,
/// bitrate-recv reads noticeably lower than bitrate-sent even on an idle link.
///
/// Without averaging, that shows up as a spurious decrease once per keyframe.
/// Averaging over a full GOP weighs the keyframe tick against the rest of the GOP.
///
/// No decision is taken until a full window has been collected after each
/// decrease, so the window is bounded to keep the controller responsive with
/// long or unbounded GOPs.
const DEFAULT_DELAY_SIGNAL_WINDOW: Duration = Duration::from_millis(1000);
/// Lower bound of how short a GOP could realistically be (1/2 a second)
const MIN_DELAY_SIGNAL_WINDOW: Duration = Duration::from_millis(500);
/// Upper bound of how long a GOP could realistically be (3 seconds)
const MAX_DELAY_SIGNAL_WINDOW: Duration = Duration::from_secs(3);

/// Delay signal window covering the longest GOP among the encoders, or the
/// default when no encoder has a known GOP (automatic or request-only
/// keyframe placement, variable framerate)
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

/// Factor the delay controller cuts by on congestion. Deep enough to drain
/// the queue a sudden capacity drop builds in one step, short drops are
/// handled by the fast recovery that follows.
const DELAY_DECREASE_FACTOR: f64 = 0.85;

/// Deepest single cut the delay controller makes when the averaged delay
/// factor says even less than DELAY_DECREASE_FACTOR is getting through
const MIN_DELAY_DECREASE_FACTOR: f64 = 0.5;

/// Decreases from one controller less than this apart belong to one
/// congestion episode
const EPISODE_GAP: Duration = Duration::from_secs(2);

/// Fast recovery: once the link looks recovered after a congestion episode,
/// climb at this rate per second...
const RECOVERY_INCREASE_PER_SEC: f64 = 1.25;

/// ... back up to this fraction of the bitrate from before the episode
const RECOVERY_TARGET_RATIO: f64 = 0.9;

/// Fast recovery is only tried within this long after the episode's last
/// decrease, so an old episode can't cause a late jump
const RECOVERY_EXPIRY: Duration = Duration::from_secs(10);

/// Averaged delay factor at or below which the link counts as recovered
/// after a delay decrease: what was sent is all getting through, the queue
/// built during the episode is draining
const RECOVERY_MAX_DELAY_FACTOR: f64 = 0.;

/// How far above the applied target either controller's target may climb.
/// The applied target is the min of the two, so whichever controller isn't
/// limiting keeps increasing: the loss controller on every TWCC feedback
/// without loss, the delay controller because the traffic it measures stays
/// below the link. Left alone that target runs away to max-bitrate, and later
/// cuts have to work their way down from there before they affect the
/// applied target.
const TARGET_HEADROOM: f64 = 1.1;

/// Framerate assumed for the additive increase when an encoder's framerate
/// is unknown (variable framerate, or not negotiated yet)
const FALLBACK_FRAMERATE: f64 = 20.;

/// Half-width of the congestion window around the bitrate EMA, in standard
/// deviations (gcc section 5.5). Below it the delay controller increases
/// multiplicatively, inside it additively, above it the window is discarded.
const CONGESTION_WINDOW_STDDEVS: f64 = 3.;

/// Lower bound on the standard deviation used for the congestion window, as
/// a fraction of the EMA, so a window seeded by one or two decreases isn't
/// nearly zero wide
const MIN_BITRATE_STDDEV_RATIO: f64 = 0.05;

/// Time without a delay decrease after which the congestion window is
/// discarded and the delay controller probes multiplicatively again. Without
/// it, a window seeded by a decrease far below capacity keeps the controller
/// in additive mode for as long as it takes to crawl out of it.
const CONGESTION_WINDOW_EXPIRY: Duration = Duration::from_secs(5);

/// Multiplicative increase per second, for both controllers (libwebrtc's
/// AIMD rate control uses ~8%/s). Applied scaled to the time since the last
/// update so it doesn't depend on how often stats or feedback arrive.
const MULTIPLICATIVE_INCREASE_PER_SEC: f64 = 1.08;

/// Lower bound of the additive increase rate, in bits per second
const MIN_ADDITIVE_INCREASE_BPS_PER_SEC: f64 = 10_000.;

/// Network response time on top of the RTT for the additive increase, as in
/// gcc section 5.5
const RESPONSE_TIME_BASE: Duration = Duration::from_millis(100);

/// Longest interval a single increase is scaled over, so a stall in the
/// stats doesn't turn into one large step
const MAX_INCREASE_INTERVAL: Duration = Duration::from_secs(1);

/// Without per-packet feedback from the pacer for this long, the controller
/// goes back to the aggregate TWCC stats
const PACKET_FEEDBACK_TIMEOUT: Duration = Duration::from_secs(2);

/// Queuing delay at the start of delta frames above which the packet-level
/// delay controller decreases...
const QUEUE_DELAY_OVERUSE: Duration = Duration::from_millis(40);

/// ... and at or below which it increases; in between it holds
const QUEUE_DELAY_NORMAL: Duration = Duration::from_millis(15);

/// On congestion the packet-level delay controller cuts to at most this
/// fraction of the capacity measured from bursts draining
const CAPACITY_TARGET_RATIO: f64 = 0.9;

/// Delta frame packets the loss controller waits for, across feedbacks,
/// before deciding on their loss
const MIN_LOSS_PACKETS: usize = 10;

/// Smoothing factor of the retransmission rate, per feedback
const RTX_RATE_ALPHA: f64 = 0.2;

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

/// A run of decreases from one controller close together in time, and the
/// fast recovery that may follow it. Cuts are deep, so a capacity drop that
/// turns out to be short (a fade, a handover, a short outage) would
/// otherwise cost tens of seconds of climbing back at the normal rate.
#[derive(Debug, Default)]
struct CongestionEpisode {
    /// The controller's target just before the episode's first decrease
    pre_bitrate: Option<f64>,
    /// When the episode's most recent decrease happened
    last_decrease: Option<Instant>,
    /// Fast recovery is underway
    recovering: bool,
    /// Congestion came back during fast recovery: the drop was real, so no
    /// second attempt in this episode
    spent: bool,
}

impl CongestionEpisode {
    /// Records a decrease of a target that was at `bitrate`. Returns true if
    /// it interrupted a fast recovery.
    fn on_decrease(&mut self, now: Instant, bitrate: f64) -> bool {
        let ongoing = self.pre_bitrate.is_some()
            && (self.recovering
                || self
                    .last_decrease
                    .is_some_and(|t| now.duration_since(t) < EPISODE_GAP));
        let interrupted = ongoing && self.recovering;
        if interrupted {
            self.recovering = false;
            self.spent = true;
        } else if !ongoing {
            *self = Self {
                pre_bitrate: Some(bitrate),
                ..Default::default()
            };
        }
        self.last_decrease = Some(now);
        interrupted
    }

    /// The fast recovery step to take instead of the controller's normal
    /// increase, if any. `bitrate` is the controller's current target,
    /// `recovered` whether its signals say the link is back. Returns the
    /// increase and whether this step starts the recovery.
    fn recovery_increase(
        &mut self,
        now: Instant,
        bitrate: f64,
        recovered: bool,
        interval_secs: f64,
    ) -> Option<(IncreaseType, bool)> {
        let pre = self.pre_bitrate?;
        if self
            .last_decrease
            .map_or(true, |t| now.duration_since(t) >= RECOVERY_EXPIRY)
        {
            *self = Self::default();
            return None;
        }
        if self.spent {
            return None;
        }
        let goal = RECOVERY_TARGET_RATIO * pre;
        if bitrate >= goal {
            // recovered, the episode is over
            *self = Self::default();
            return None;
        }
        let starting = !self.recovering;
        if starting {
            if !recovered {
                return None;
            }
            self.recovering = true;
        }
        let factor = f64::min(
            RECOVERY_INCREASE_PER_SEC.powf(interval_secs),
            goal / bitrate,
        );
        Some((IncreaseType::FastRecovery(factor), starting))
    }
}

/// Sliding-window average of the delay controller's input signals.
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
    /// samples have been collected for a full window since the last reset.
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
    /// Increase bitrate by factor, recovering from a congestion episode
    /// that turned out to be short; its own variant so it shows in the logs
    FastRecovery(f64),
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
    /// Time of the last delay controller update, increases are scaled to
    /// the time elapsed since
    last_update_time: Option<std::time::Instant>,
    /// Time of the last loss controller update, same purpose
    last_loss_update_time: Option<std::time::Instant>,
    /// Time of the last delay decrease, used to expire the congestion window
    last_delay_decrease: Option<std::time::Instant>,
    /// Congestion episodes and fast recovery, per controller
    delay_episode: CongestionEpisode,
    loss_episode: CongestionEpisode,
    /// Delay signals averaged over a GOP-long window before thresholding,
    /// when the controller runs on the aggregate TWCC stats
    delay_signals: DelaySignals,
    /// Paces the stream at the applied target, and gives the per-packet
    /// feedback the controller runs on when it has it
    pacer: Option<gst::Element>,
    /// Congestion signals from the per-packet feedback
    detector: PacketDetector,
    /// Last time per-packet feedback came in
    last_packet_feedback: Option<Instant>,
    /// Whether the controller ran on per-packet feedback last time it
    /// decided, to log when it switches
    on_packet_feedback: bool,
    /// Delta frame packets reported, and lost, since the loss controller's
    /// last decision
    delta_packets: usize,
    delta_lost: usize,
    /// Smoothed rate of retransmissions, bit/sec: they share the link with
    /// the encoders, so the encoders get the target minus this
    rtx_rate: f64,
    /// Latest round-trip time from the stats, seconds
    rtt: f64,
    /// For logging purposes
    peer_id: String,

    min_bitrate: u32,
    max_bitrate: u32,
    do_fec: bool,
}

impl CongestionController {
    pub fn new(
        peer_id: &str,
        min_bitrate: u32,
        max_bitrate: u32,
        do_fec: bool,
        pacer: Option<gst::Element>,
    ) -> Self {
        Self {
            target_bitrate_on_delay: 0,
            target_bitrate_on_loss: 0,
            bitrate_ema: None,
            bitrate_emvar: 0.,
            last_update_time: None,
            last_loss_update_time: None,
            last_delay_decrease: None,
            delay_episode: CongestionEpisode::default(),
            loss_episode: CongestionEpisode::default(),
            delay_signals: DelaySignals::default(),
            pacer,
            detector: PacketDetector::default(),
            last_packet_feedback: None,
            on_packet_feedback: false,
            delta_packets: 0,
            delta_lost: 0,
            rtx_rate: 0.,
            rtt: 0.,
            peer_id: peer_id.to_string(),
            min_bitrate,
            max_bitrate,
            do_fec,
        }
    }

    /// Whether the controller runs on the pacer's per-packet feedback
    /// rather than the aggregate TWCC stats
    fn packet_level(&self) -> bool {
        self.last_packet_feedback
            .is_some_and(|t| t.elapsed() < PACKET_FEEDBACK_TIMEOUT)
    }

    fn update_delay(
        &mut self,
        element: &super::BaseWebRTCSink,
        twcc_stats: &gst::StructureRef,
        rtt: f64,
        encoders: &[VideoEncoder],
    ) -> CongestionControlOp {
        // Unwrap, all those fields must be there or there's been an API
        // break, which qualifies as programming error
        let bitrate_sent = twcc_stats.get::<u32>("bitrate-sent").unwrap();
        let bitrate_recv = twcc_stats.get::<u32>("bitrate-recv").unwrap();
        let delta_of_delta = twcc_stats.get::<i64>("avg-delta-of-delta").unwrap();

        let now = Instant::now();
        let interval_secs = increase_interval_secs(self.last_update_time.replace(now), now);

        let delay_duration = delay_signal_window(encoders);
        if self.delay_signals.set_window(delay_duration) {
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

        // Signed on purpose: burst-drain artifacts push recv both above and
        // below sent, and only a sustained excess of sent over recv means
        // congestion. Clamping each sample to >= 0 (saturating_sub) before
        // averaging would turn that zero-mean noise into a positive bias.
        if bitrate_sent > 0 {
            // Normalized by `bitrate_sent` instead of `target_bitrate`
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
            // delay_factor is the share of what was sent that isn't getting
            // through, so cut at least down to what is
            let through = 1. - delay_factor;
            let (factor, reason) = if through >= DELAY_DECREASE_FACTOR {
                (
                    DELAY_DECREASE_FACTOR,
                    format!("low delay factor {delay_factor}"),
                )
            } else {
                (
                    through.max(MIN_DELAY_DECREASE_FACTOR),
                    format!("High delay factor {delay_factor}"),
                )
            };

            self.delay_signals.reset();
            CongestionControlOp::Decrease { factor, reason }
        } else if delta_of_delta > DELTA_OF_DELTA_THRESHOLD_NS {
            self.delay_signals.reset();
            CongestionControlOp::Decrease {
                factor: DELAY_DECREASE_FACTOR,
                reason: format!("High delta: {delta_of_delta}"),
            }
        } else {
            self.delay_increase(
                element,
                now,
                interval_secs,
                rtt,
                delay_factor <= RECOVERY_MAX_DELAY_FACTOR,
                encoders,
            )
        }
    }

    /// The delay controller's increase once its signals are clear.
    /// `recovered` tells whether they say the link is back after a
    /// congestion episode, for fast recovery.
    fn delay_increase(
        &mut self,
        element: &super::BaseWebRTCSink,
        now: Instant,
        interval_secs: f64,
        rtt: f64,
        recovered: bool,
        encoders: &[VideoEncoder],
    ) -> CongestionControlOp {
        let target_bitrate = f64::min(
            self.target_bitrate_on_delay as f64,
            self.target_bitrate_on_loss as f64,
        );

        if let Some((increase, starting)) = self.delay_episode.recovery_increase(
            now,
            self.target_bitrate_on_delay as f64,
            recovered,
            interval_secs,
        ) {
            if starting {
                gst::info!(
                    CAT,
                    obj = element,
                    "consumer {}: delay signals clear, fast recovery towards {:.0}",
                    self.peer_id,
                    RECOVERY_TARGET_RATIO * self.delay_episode.pre_bitrate.unwrap_or(0.),
                );
            }
            return CongestionControlOp::Increase(increase);
        }

        if self.bitrate_ema.is_some()
            && self
                .last_delay_decrease
                .is_some_and(|t| now.duration_since(t) >= CONGESTION_WINDOW_EXPIRY)
        {
            gst::trace!(
                CAT,
                obj = element,
                "consumer {}: no decrease for {:?}, discarding congestion window",
                self.peer_id,
                CONGESTION_WINDOW_EXPIRY
            );
            self.bitrate_ema = None;
        }

        CongestionControlOp::Increase(if let Some(ema) = self.bitrate_ema {
            let bitrate_stdev = f64::max(self.bitrate_emvar.sqrt(), MIN_BITRATE_STDDEV_RATIO * ema);

            gst::trace!(
                CAT,
                obj = element,
                "consumer {}: Old bitrate: {}, ema: {}, stddev: {}",
                self.peer_id,
                target_bitrate,
                ema,
                bitrate_stdev,
            );

            if target_bitrate < ema - CONGESTION_WINDOW_STDDEVS * bitrate_stdev {
                gst::trace!(
                    CAT,
                    obj = element,
                    "consumer {}: below last congestion window",
                    self.peer_id
                );
                /* Multiplicative increase */
                multiplicative_increase(interval_secs)
            } else if target_bitrate > ema + CONGESTION_WINDOW_STDDEVS * bitrate_stdev {
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
            } else {
                // About one packet per response time, as in gcc section 5.5
                // and libwebrtc. The response time is how long the network
                // takes to show the effect of an increase, not how long the
                // detector averages for, so it doesn't include the delay
                // signal window.
                let response_time_secs = RESPONSE_TIME_BASE.as_secs_f64() + rtt;
                let avg_packet_size_bits =
                    self.avg_packet_size_bits(element, target_bitrate, encoders);
                let rate_bps_per_sec = f64::max(
                    MIN_ADDITIVE_INCREASE_BPS_PER_SEC,
                    avg_packet_size_bits / response_time_secs,
                );

                gst::trace!(
                    CAT,
                    obj = element,
                    "consumer {}: still in last congestion window, increasing by {:.0} bps/s",
                    self.peer_id,
                    rate_bps_per_sec,
                );

                /* Additive increase */
                IncreaseType::Additive(rate_bps_per_sec * interval_secs)
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

    /// Expected average RTP packet size in bits, averaged over the encoders.
    /// target_bitrate is the total for all encoders and is split equally,
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
                let mtu_bits = f64::from(encoder.mtu()) * 8.;
                let bits_per_frame = bitrate_per_encoder / framerate;
                let packets_per_frame = f64::max(1., f64::ceil(bits_per_frame / mtu_bits));

                gst::trace!(
                    CAT,
                    obj = element,
                    "consumer {}: framerate {}, mtu {}, {} packets per frame",
                    self.peer_id,
                    framerate,
                    encoder.mtu(),
                    packets_per_frame
                );

                bits_per_frame / packets_per_frame
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
        // The per-packet feedback tells delta frame losses apart, when
        // there is some
        if self.packet_level() {
            return;
        }

        let loss_percentage = stats.get::<f64>("packet-loss-pct").unwrap();
        let op = self.loss_op(element, loss_percentage);
        self.apply_control_op(element, encoders, op, ControllerType::Loss);
    }

    fn loss_op(
        &mut self,
        element: &super::BaseWebRTCSink,
        loss_percentage: f64,
    ) -> CongestionControlOp {
        let now = Instant::now();
        let interval_secs = increase_interval_secs(self.last_loss_update_time.replace(now), now);

        if loss_percentage > 10. {
            CongestionControlOp::Decrease {
                factor: ((100. - (0.5 * loss_percentage)) / 100.).clamp(0.7, 0.98),
                reason: format!("High loss: {loss_percentage}"),
            }
        } else if loss_percentage > 2. {
            CongestionControlOp::Hold
        } else {
            // Loss is low, so the link is back as far as this controller can
            // tell: fast recovery applies right away
            CongestionControlOp::Increase(
                match self.loss_episode.recovery_increase(
                    now,
                    self.target_bitrate_on_loss as f64,
                    true,
                    interval_secs,
                ) {
                    Some((increase, starting)) => {
                        if starting {
                            gst::info!(
                                CAT,
                                obj = element,
                                "consumer {}: loss down to {:.1}%, fast recovery towards {:.0}",
                                self.peer_id,
                                loss_percentage,
                                RECOVERY_TARGET_RATIO * self.loss_episode.pre_bitrate.unwrap_or(0.),
                            );
                        }
                        increase
                    }
                    None => multiplicative_increase(interval_secs),
                },
            )
        }
    }

    pub fn delay_control(
        &mut self,
        element: &super::BaseWebRTCSink,
        stats: &gst::StructureRef,
        encoders: &mut [VideoEncoder],
    ) {
        self.rtt = self.lookup_rtt(stats);

        // The stats still give the RTT, but the per-packet feedback is the
        // better delay signal when there is some
        if self.packet_level() {
            return;
        }
        if self.on_packet_feedback {
            self.on_packet_feedback = false;
            gst::info!(
                CAT,
                obj = element,
                "consumer {}: no per-packet feedback for {:?}, back on aggregate TWCC stats",
                self.peer_id,
                PACKET_FEEDBACK_TIMEOUT
            );
        }

        if let Some(twcc_stats) = lookup_twcc_stats(stats) {
            let op = self.update_delay(element, &twcc_stats, self.rtt, encoders);
            self.apply_control_op(element, encoders, op, ControllerType::Delay);
        }
    }

    /// Runs both controllers on the pacer's per-packet feedback (an
    /// `RTPPacerFeedback` structure)
    pub fn packet_feedback(
        &mut self,
        element: &super::BaseWebRTCSink,
        feedback: &gst::StructureRef,
        encoders: &mut [VideoEncoder],
    ) {
        self.update_pacer_params(encoders);

        let packets = packet_feedback::parse(feedback);
        let unknown = packets
            .iter()
            .filter(|p| p.kind == PacketKind::Unknown)
            .count();
        // Until the pacer lines the feedback up with what it sent, it can't
        // tell what the packets were
        if unknown == packets.len() {
            gst::log!(
                CAT,
                obj = element,
                "consumer {}: pacer can't tell what the {} packets of this feedback were, skipping it",
                self.peer_id,
                packets.len()
            );
            return;
        }
        if unknown > 0 {
            gst::debug!(
                CAT,
                obj = element,
                "consumer {}: {unknown} of {} feedback packets unknown to the pacer, \
                 left out of the signals",
                self.peer_id,
                packets.len()
            );
        }

        let now = Instant::now();
        if !self.on_packet_feedback {
            self.on_packet_feedback = true;
            gst::info!(
                CAT,
                obj = element,
                "consumer {}: running on per-packet feedback",
                self.peer_id
            );
        }
        self.last_packet_feedback = Some(now);

        let signals = self.detector.update(&packets);
        if signals.send_span_ns > 0 {
            let rtx_rate = signals.rtx_bits as f64 / (signals.send_span_ns as f64 / 1e9);
            self.rtx_rate += RTX_RATE_ALPHA * (rtx_rate - self.rtx_rate);
        }

        gst::trace!(
            CAT,
            obj = element,
            "consumer {}: queue delay {:?} over {} delta frames, drain overrun {}, capacity {:?}, \
             delta loss {}/{}, burst loss {}/{}, rtx rate {:.0}",
            self.peer_id,
            signals.queue_delay,
            signals.delay_samples,
            signals.drain_overrun,
            signals.capacity,
            signals.delta_lost,
            signals.delta_packets,
            signals.burst_lost,
            signals.burst_packets,
            self.rtx_rate,
        );

        if signals.burst_lost > 0 && signals.delta_lost == 0 && signals.delta_packets > 0 {
            gst::debug!(
                CAT,
                obj = element,
                "consumer {}: {} keyframe/RTX packets lost with delta frames intact, \
                 bursts overflowing the bottleneck buffer",
                self.peer_id,
                signals.burst_lost,
            );
        }

        let op = self.packet_delay_op(element, &signals, now, encoders);
        self.apply_control_op(element, encoders, op, ControllerType::Delay);

        // At low bitrates a feedback can hold only a packet or two
        self.delta_packets += signals.delta_packets;
        self.delta_lost += signals.delta_lost;
        if self.delta_packets >= MIN_LOSS_PACKETS {
            let loss_percentage = 100. * self.delta_lost as f64 / self.delta_packets as f64;
            if self.delta_lost > 0 {
                gst::debug!(
                    CAT,
                    obj = element,
                    "consumer {}: delta frame loss {loss_percentage:.1}% ({} of {} packets)",
                    self.peer_id,
                    self.delta_lost,
                    self.delta_packets,
                );
            }
            self.delta_packets = 0;
            self.delta_lost = 0;
            let op = self.loss_op(element, loss_percentage);
            self.apply_control_op(element, encoders, op, ControllerType::Loss);
        }
    }

    /// The delay controller on the per-packet signals: the queue standing at
    /// the start of delta frames, and bursts that drain slower than their
    /// own size explains. Keyframes and retransmissions are left out.
    fn packet_delay_op(
        &mut self,
        element: &super::BaseWebRTCSink,
        signals: &Signals,
        now: Instant,
        encoders: &[VideoEncoder],
    ) -> CongestionControlOp {
        let interval_secs = increase_interval_secs(self.last_update_time.replace(now), now);

        let Some(queue_delay) = signals.queue_delay else {
            gst::trace!(
                CAT,
                obj = element,
                "consumer {}: no delta frame delay sample yet, holding",
                self.peer_id
            );
            return CongestionControlOp::Hold;
        };

        // The queue needs a response time to drain after a cut, and the
        // first packets of at least two frames to show it
        let framerate = encoders
            .iter()
            .filter_map(VideoEncoder::framerate)
            .reduce(f64::min)
            .unwrap_or(FALLBACK_FRAMERATE);
        let settle = f64::max(RESPONSE_TIME_BASE.as_secs_f64() + self.rtt, 2. / framerate);
        let settling = self
            .last_delay_decrease
            .is_some_and(|t| now.duration_since(t).as_secs_f64() < settle);

        if signals.drain_overrun || queue_delay > QUEUE_DELAY_OVERUSE {
            if settling {
                gst::debug!(
                    CAT,
                    obj = element,
                    "consumer {}: congested (queue delay {:?}, drain overrun {}) but still \
                     settling after the last decrease ({:.0} ms), holding",
                    self.peer_id,
                    queue_delay,
                    signals.drain_overrun,
                    settle * 1000.,
                );
                return CongestionControlOp::Hold;
            }

            let current = self.target_bitrate_on_delay as f64;
            let mut factor = DELAY_DECREASE_FACTOR;
            if let Some(capacity) = signals.capacity {
                factor = factor.min(CAPACITY_TARGET_RATIO * capacity / current);
            }
            let factor = factor.max(MIN_DELAY_DECREASE_FACTOR);

            gst::debug!(
                CAT,
                obj = element,
                "consumer {}: congested, queue delay {:?} (overuse above {:?}), drain overrun {}, \
                 capacity {:?}: cutting delay target {:.0} by {:.3}",
                self.peer_id,
                queue_delay,
                QUEUE_DELAY_OVERUSE,
                signals.drain_overrun,
                signals.capacity,
                current,
                factor,
            );

            self.detector.reset_queue_delay();
            return CongestionControlOp::Decrease {
                factor,
                reason: if signals.drain_overrun {
                    format!("burst drained slower than capacity {:?}", signals.capacity)
                } else {
                    format!("queue delay {queue_delay:?}")
                },
            };
        }

        if settling || queue_delay > QUEUE_DELAY_NORMAL {
            gst::trace!(
                CAT,
                obj = element,
                "consumer {}: holding, queue delay {:?} (increase at or below {:?}), settling {}",
                self.peer_id,
                queue_delay,
                QUEUE_DELAY_NORMAL,
                settling,
            );
            return CongestionControlOp::Hold;
        }

        self.delay_increase(element, now, interval_secs, self.rtt, true, encoders)
    }

    /// Tells the pacer the framerate and GOP length the encoders run at
    fn update_pacer_params(&self, encoders: &[VideoEncoder]) {
        let Some(pacer) = self.pacer.as_ref() else {
            return;
        };

        if let Some(framerate) = encoders
            .iter()
            .filter_map(VideoEncoder::framerate)
            .reduce(f64::min)
        {
            if (pacer.property::<f64>("framerate") - framerate).abs() > 0.01 {
                gst::debug!(
                    CAT,
                    obj = pacer,
                    "consumer {}: encoders run at {framerate} fps, telling the pacer",
                    self.peer_id
                );
                pacer.set_property("framerate", framerate.clamp(1., 1000.));
            }
        }

        if let Some(gop_size) = encoders
            .iter()
            .filter_map(VideoEncoder::keyframe_interval)
            .max()
        {
            let gop_size = u32::try_from(gop_size).unwrap_or(u32::MAX);
            if pacer.property::<u32>("gop-size") != gop_size {
                gst::debug!(
                    CAT,
                    obj = pacer,
                    "consumer {}: encoders' GOP is {gop_size} frames, telling the pacer",
                    self.peer_id
                );
                pacer.set_property("gop-size", gop_size);
            }
        }
    }

    /// Called when encoders are added
    pub fn encoders_changed(&self, encoders: &[VideoEncoder]) {
        self.update_pacer_params(encoders);
        if let Some(pacer) = self.pacer.as_ref() {
            let target = i32::min(self.target_bitrate_on_delay, self.target_bitrate_on_loss);
            if target > 0 {
                pacer.set_property("bitrate", target as u32);
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
        // Keeps increases within TARGET_HEADROOM of the applied target. This
        // can also pull a target down, when the other controller has lowered
        // the applied target since.
        let cap_increase =
            |bitrate: i32| i32::min(bitrate, (prev_bitrate as f64 * TARGET_HEADROOM) as i32);
        match &control_op {
            CongestionControlOp::Hold => {}
            CongestionControlOp::Increase(IncreaseType::Additive(value)) => {
                self.clamp_bitrate(
                    cap_increase(current_bitrate + *value as i32),
                    n_encoders,
                    controller_type,
                );
            }
            CongestionControlOp::Increase(
                IncreaseType::Multiplicative(factor) | IncreaseType::FastRecovery(factor),
            ) => {
                self.clamp_bitrate(
                    cap_increase((current_bitrate as f64 * factor) as i32),
                    n_encoders,
                    controller_type,
                );
            }
            CongestionControlOp::Decrease { factor, .. } => {
                self.clamp_bitrate(
                    (current_bitrate as f64 * factor) as i32,
                    n_encoders,
                    controller_type,
                );

                let now = Instant::now();
                let episode = match controller_type {
                    ControllerType::Delay => &mut self.delay_episode,
                    ControllerType::Loss => &mut self.loss_episode,
                };
                if episode.on_decrease(now, current_bitrate as f64) {
                    gst::info!(
                        CAT,
                        obj = element,
                        "consumer {}: {:?} congestion came back during fast recovery, capacity drop is real",
                        self.peer_id,
                        controller_type,
                    );
                }

                if let ControllerType::Delay = controller_type {
                    self.last_delay_decrease = Some(now);

                    // The EMA tracks the bitrate at which congestion showed up,
                    // so it's fed the target from before the cut: the cut
                    // target is below the capacity, and a window centred on
                    // it would sit below the capacity too
                    let congested_at = current_bitrate as f64;
                    // Smoothing factor
                    let alpha = 0.75;
                    if let Some(ema) = self.bitrate_ema {
                        let sigma: f64 = congested_at - ema;
                        self.bitrate_ema = Some(ema + (alpha * sigma));
                        self.bitrate_emvar =
                            (1. - alpha) * (self.bitrate_emvar + alpha * sigma.powi(2));
                    } else {
                        self.bitrate_ema = Some(congested_at);
                        self.bitrate_emvar = 0.;
                    }
                }
            }
        }

        let min_total = self.min_bitrate as i32 * n_encoders;
        let total_bitrate = i32::min(self.target_bitrate_on_delay, self.target_bitrate_on_loss)
            .clamp(min_total, self.max_bitrate as i32 * n_encoders);
        // Retransmissions share the link with the encoders
        let encoders_bitrate = f64::max(total_bitrate as f64 - self.rtx_rate, min_total as f64);
        let target_bitrate = encoders_bitrate as i32 / n_encoders;

        if total_bitrate != prev_bitrate {
            if let Some(pacer) = self.pacer.as_ref() {
                pacer.set_property("bitrate", total_bitrate.max(1) as u32);
            }

            if self.rtx_rate >= 1. {
                gst::debug!(
                    CAT,
                    obj = element,
                    "consumer {}: encoders get {:.0} in total: target {} minus {:.0} of retransmissions",
                    self.peer_id,
                    encoders_bitrate,
                    total_bitrate,
                    self.rtx_rate,
                );
            }

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

#[cfg(test)]
mod tests {
    use super::*;

    fn factor(step: Option<(IncreaseType, bool)>) -> Option<f64> {
        match step {
            Some((IncreaseType::FastRecovery(f), _)) => Some(f),
            _ => None,
        }
    }

    #[test]
    fn short_drop_recovers_fast_up_to_goal() {
        let t0 = Instant::now();
        let mut e = CongestionEpisode::default();
        assert!(!e.on_decrease(t0, 3_000_000.));
        // a second cut within EPISODE_GAP is the same episode
        assert!(!e.on_decrease(t0 + Duration::from_secs(1), 2_550_000.));
        assert_eq!(e.pre_bitrate, Some(3_000_000.));

        // no recovery until the signals say the link is back
        let t = t0 + Duration::from_secs(2);
        assert!(e.recovery_increase(t, 2_170_000., false, 0.1).is_none());

        let (increase, starting) = e.recovery_increase(t, 2_170_000., true, 0.1).unwrap();
        assert!(starting);
        let IncreaseType::FastRecovery(f) = increase else {
            panic!("expected fast recovery, got {increase:?}");
        };
        assert!((f - RECOVERY_INCREASE_PER_SEC.powf(0.1)).abs() < 1e-9);

        // once underway it doesn't need the signal again, and never
        // overshoots the goal
        let f = factor(e.recovery_increase(t, 2_690_000., false, 1.)).unwrap();
        assert!((2_690_000. * f - RECOVERY_TARGET_RATIO * 3_000_000.).abs() < 1.);

        // reaching the goal ends the episode
        assert!(e.recovery_increase(t, 2_700_000., true, 0.1).is_none());
        assert_eq!(e.pre_bitrate, None);
    }

    #[test]
    fn congestion_during_recovery_ends_it() {
        let t0 = Instant::now();
        let mut e = CongestionEpisode::default();
        e.on_decrease(t0, 3_000_000.);
        let t = t0 + Duration::from_secs(3);
        assert!(factor(e.recovery_increase(t, 2_550_000., true, 0.1)).is_some());

        // more than EPISODE_GAP after the last cut, but recovering: same
        // episode, and the drop turned out to be real
        assert!(e.on_decrease(t + Duration::from_secs(1), 2_600_000.));
        assert!(e
            .recovery_increase(t + Duration::from_secs(2), 2_200_000., true, 0.1)
            .is_none());
    }

    #[test]
    fn recovery_expires() {
        let t0 = Instant::now();
        let mut e = CongestionEpisode::default();
        e.on_decrease(t0, 3_000_000.);
        assert!(e
            .recovery_increase(t0 + RECOVERY_EXPIRY, 2_550_000., true, 0.1)
            .is_none());
        assert_eq!(e.pre_bitrate, None);

        // a later decrease starts a new episode from its own bitrate
        let t1 = t0 + RECOVERY_EXPIRY + Duration::from_secs(1);
        assert!(!e.on_decrease(t1, 1_000_000.));
        assert_eq!(e.pre_bitrate, Some(1_000_000.));
    }
}
