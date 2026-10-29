// SPDX-License-Identifier: MPL-2.0

//! Per-packet congestion signals from the `rtppacer` feedback.
//!
//! The pacer tells, for every packet of the TWCC feedback, what it was: part
//! of a keyframe (or of a frame sized like one), part of a delta frame, or a
//! retransmission. That lets the detector separate the queue the sender
//! builds itself from the queue the network builds:
//!
//! - A burst (keyframe, large frame, RTX) and the packets queued right
//!   behind it at the bottleneck drain at the link rate. Those packets are
//!   left out of the delay signal. Their arrival spread measures the
//!   capacity, and a burst that takes much longer to drain than its own
//!   bytes explain is congestion.
//! - Outside of bursts, the queuing delay of the first packet of each delta
//!   frame is the queue that was standing before the frame was sent, which
//!   only builds up when the link can't keep up.
//! - Losses are counted on delta frames. Lost keyframe or RTX packets with
//!   clean delta frames are the sender's bursts overflowing a shallow
//!   buffer, which cutting the bitrate doesn't help.

use once_cell::sync::Lazy;
use std::collections::VecDeque;
use std::time::Duration;

static CAT: Lazy<gst::DebugCategory> = Lazy::new(|| {
    gst::DebugCategory::new(
        "webrtcsink-packetfeedback",
        gst::DebugColorFlags::empty(),
        Some("WebRTC sink per-packet congestion signals"),
    )
});

fn ms(ns: impl Into<f64>) -> f64 {
    ns.into() / 1e6
}

/// Window over which the smallest one-way delay is taken as the delay of
/// an empty queue
const BASE_DELAY_WINDOW_NS: i64 = 10_000_000_000;
const BASE_DELAY_BUCKET_NS: i64 = 1_000_000_000;

/// Smoothing factor of the queuing delay, per delta frame
const QUEUE_DELAY_ALPHA: f64 = 0.25;

/// Packets arriving at most this far apart (or twice their transmission
/// time at the estimated capacity, if longer) were queued back to back
const BACK_TO_BACK_MIN_NS: i64 = 1_000_000;

/// A burst may take this many times the time its bytes need at the
/// estimated capacity to drain...
const DRAIN_MARGIN: f64 = 1.5;
/// ... plus this, before the queue behind it counts as congestion
const DRAIN_SLACK_NS: f64 = 20_000_000.;
/// Without a capacity estimate, a burst draining for longer than this is
/// congestion
const MAX_DRAIN_NS: i64 = 2_000_000_000;

/// A drain spanning at least this many bytes and this long measures the
/// capacity
const MIN_CAPACITY_SAMPLE_BYTES: u64 = 10_000;
const MIN_CAPACITY_SAMPLE_NS: i64 = 2_000_000;
/// The drain was limited by the bottleneck, not by the pacer, when it
/// arrived this much more spread out than it was sent
const QUEUE_LIMITED_SPREAD: f64 = 1.1;
const CAPACITY_ALPHA: f64 = 0.3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PacketKind {
    Media,
    RtxRepair,
    RtxPadding,
    /// The pacer couldn't tell what the packet was
    Unknown,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct FeedbackPacket {
    pub send_ns: i64,
    pub arrival_ns: Option<i64>,
    /// RTP payload size
    pub size: u32,
    pub kind: PacketKind,
    pub keyframe: bool,
    pub large_frame: bool,
    /// (ssrc, RTP timestamp) of the frame, for media
    pub frame: Option<(u32, u32)>,
}

impl FeedbackPacket {
    /// Sent as part of a burst: a keyframe, a frame sized like one, or a
    /// retransmission
    pub fn is_burst(&self) -> bool {
        match self.kind {
            PacketKind::RtxRepair | PacketKind::RtxPadding => true,
            PacketKind::Media => self.keyframe || self.large_frame,
            PacketKind::Unknown => false,
        }
    }

    fn is_delta(&self) -> bool {
        self.kind == PacketKind::Media && !self.is_burst()
    }

    fn from_structure(s: &gst::StructureRef) -> Option<Self> {
        let kind = match s.get::<&str>("kind").ok()? {
            "media" => PacketKind::Media,
            "rtx-repair" => PacketKind::RtxRepair,
            "rtx-padding" => PacketKind::RtxPadding,
            _ => PacketKind::Unknown,
        };
        let frame = s
            .get::<u32>("ssrc")
            .ok()
            .zip(s.get::<u32>("rtp-timestamp").ok())
            .filter(|_| kind == PacketKind::Media);

        Some(Self {
            send_ns: s.get::<u64>("local-ts").ok()? as i64,
            arrival_ns: s.get::<u64>("remote-ts").ok().map(|ts| ts as i64),
            size: s.get::<u32>("size").ok()?,
            kind,
            keyframe: s.get::<bool>("keyframe").unwrap_or(false),
            large_frame: s.get::<bool>("large-frame").unwrap_or(false),
            frame,
        })
    }
}

/// Parses an `RTPPacerFeedback` structure, packets in TWCC order
pub(crate) fn parse(feedback: &gst::StructureRef) -> Vec<FeedbackPacket> {
    let Ok(packets) = feedback.get::<gst::ArrayRef>("packets") else {
        gst::warning!(CAT, "Feedback without packets: {feedback}");
        return Vec::new();
    };

    let parsed = packets
        .iter()
        .filter_map(|v| v.get::<gst::Structure>().ok())
        .filter_map(|s| FeedbackPacket::from_structure(&s))
        .collect::<Vec<_>>();
    if parsed.len() != packets.len() {
        gst::debug!(
            CAT,
            "{} of {} feedback packets couldn't be parsed",
            packets.len() - parsed.len(),
            packets.len()
        );
    }

    parsed
}

/// What one feedback says about the network
#[derive(Debug, Default, Clone)]
pub(crate) struct Signals {
    /// Smoothed queuing delay at the start of delta frames, None until the
    /// first one
    pub queue_delay: Option<Duration>,
    /// Number of delta frames that updated it
    pub delay_samples: usize,
    /// A burst took longer to drain than its own bytes explain
    pub drain_overrun: bool,
    /// Bottleneck capacity measured from bursts draining, bit/sec
    pub capacity: Option<f64>,
    pub delta_packets: usize,
    pub delta_lost: usize,
    pub burst_packets: usize,
    pub burst_lost: usize,
    /// Bits sent as retransmissions, and over how long the feedback's
    /// packets were sent
    pub rtx_bits: u64,
    pub send_span_ns: i64,
}

#[derive(Debug)]
struct Drain {
    start_arrival: i64,
    last_arrival: i64,
    first_send: i64,
    last_send: i64,
    first_size: u32,
    bytes: u64,
}

impl Drain {
    fn new(p: &FeedbackPacket, arrival: i64) -> Self {
        Self {
            start_arrival: arrival,
            last_arrival: arrival,
            first_send: p.send_ns,
            last_send: p.send_ns,
            first_size: p.size,
            bytes: u64::from(p.size),
        }
    }

    fn add(&mut self, p: &FeedbackPacket, arrival: i64) {
        self.last_arrival = self.last_arrival.max(arrival);
        self.last_send = self.last_send.max(p.send_ns);
        self.bytes += u64::from(p.size);
    }

    /// Whether, at `arrival`, the queue behind the burst has been draining
    /// for longer than the burst explains
    fn overdue(&self, arrival: i64, capacity: Option<f64>) -> bool {
        let elapsed = arrival - self.start_arrival;
        match self.expected_ns(capacity) {
            Some(expected) => elapsed as f64 > expected * DRAIN_MARGIN + DRAIN_SLACK_NS,
            None => elapsed > MAX_DRAIN_NS,
        }
    }

    /// How long the bytes of the burst and of what queued behind it take
    /// at `capacity`
    fn expected_ns(&self, capacity: Option<f64>) -> Option<f64> {
        capacity.map(|capacity| self.bytes as f64 * 8. / capacity * 1e9)
    }

    fn log_overrun(&self, arrival: i64, capacity: Option<f64>, what: &str) {
        gst::debug!(
            CAT,
            "{what}: draining for {:.1} ms, its {} bytes need {:?} ms at capacity {:?}",
            ms((arrival - self.start_arrival) as f64),
            self.bytes,
            self.expected_ns(capacity).map(ms),
            capacity,
        );
    }
}

/// Smallest one-way delay over BASE_DELAY_WINDOW_NS, in buckets so old
/// minimums age out
#[derive(Debug, Default)]
struct BaseDelay {
    buckets: VecDeque<(i64, i64)>,
}

impl BaseDelay {
    fn update(&mut self, send_ns: i64, owd: i64) {
        match self.buckets.back_mut() {
            Some((start, min)) if send_ns - *start < BASE_DELAY_BUCKET_NS => *min = (*min).min(owd),
            _ => self.buckets.push_back((send_ns, owd)),
        }
        while self
            .buckets
            .front()
            .is_some_and(|(start, _)| send_ns - *start > BASE_DELAY_WINDOW_NS)
        {
            self.buckets.pop_front();
        }
    }

    fn get(&self) -> Option<i64> {
        self.buckets.iter().map(|(_, min)| *min).min()
    }
}

#[derive(Debug, Default)]
pub(crate) struct PacketDetector {
    base_delay: BaseDelay,
    capacity: Option<f64>,
    drain: Option<Drain>,
    last_arrival: Option<i64>,
    last_delta_frame: Option<(u32, u32)>,
    queue_delay_ns: Option<f64>,
}

impl PacketDetector {
    #[cfg(test)]
    pub fn capacity(&self) -> Option<f64> {
        self.capacity
    }

    /// Processes the packets of one feedback, in TWCC order
    pub fn update(&mut self, packets: &[FeedbackPacket]) -> Signals {
        let mut signals = Signals::default();

        if let (Some(first), Some(last)) = (
            packets.iter().map(|p| p.send_ns).min(),
            packets.iter().map(|p| p.send_ns).max(),
        ) {
            signals.send_span_ns = last - first;
        }

        for p in packets {
            if p.is_burst() {
                signals.burst_packets += 1;
                signals.burst_lost += usize::from(p.arrival_ns.is_none());
            } else if p.is_delta() {
                signals.delta_packets += 1;
                signals.delta_lost += usize::from(p.arrival_ns.is_none());
            }
            if matches!(p.kind, PacketKind::RtxRepair | PacketKind::RtxPadding) {
                signals.rtx_bits += u64::from(p.size) * 8;
            }

            let Some(arrival) = p.arrival_ns else {
                continue;
            };
            let owd = arrival - p.send_ns;
            self.base_delay.update(p.send_ns, owd);
            let gap = self.last_arrival.map(|last| arrival - last);
            self.last_arrival = Some(self.last_arrival.map_or(arrival, |last| last.max(arrival)));

            if p.is_burst() {
                match self.drain.as_mut() {
                    Some(drain) => drain.add(p, arrival),
                    None => {
                        gst::trace!(
                            CAT,
                            "burst starts draining with a {:?} packet (keyframe {}, large frame {})",
                            p.kind,
                            p.keyframe,
                            p.large_frame,
                        );
                        self.drain = Some(Drain::new(p, arrival));
                    }
                }
                continue;
            }

            if let Some(drain) = self.drain.as_mut() {
                let transmission_ns = self.capacity.map_or(0, |capacity| {
                    (f64::from(p.size) * 8. / capacity * 1e9) as i64
                });
                let back_to_back =
                    gap.is_some_and(|gap| gap <= BACK_TO_BACK_MIN_NS.max(2 * transmission_ns));
                let overdue = drain.overdue(arrival, self.capacity);

                if back_to_back && !overdue {
                    // Queued behind the burst: says nothing about the network
                    drain.add(p, arrival);
                    continue;
                }

                if back_to_back {
                    drain.log_overrun(
                        arrival,
                        self.capacity,
                        "delta frames still queued behind a burst",
                    );
                    signals.drain_overrun = true;
                }
                let drain = self.drain.take().unwrap();
                self.measure_capacity(&drain);
            }

            if p.is_delta() && p.frame != self.last_delta_frame {
                // First packet of a delta frame to arrive
                self.last_delta_frame = p.frame;
                if let Some(base) = self.base_delay.get() {
                    let queue_delay = (owd - base) as f64;
                    let smoothed = match self.queue_delay_ns {
                        Some(avg) => avg + QUEUE_DELAY_ALPHA * (queue_delay - avg),
                        None => queue_delay,
                    };
                    self.queue_delay_ns = Some(smoothed);
                    signals.delay_samples += 1;
                    gst::trace!(
                        CAT,
                        "delta frame {:?}: queuing delay {:.2} ms (one-way {:.2} ms over a base \
                         of {:.2} ms), smoothed {:.2} ms",
                        p.frame,
                        ms(queue_delay),
                        ms(owd as f64),
                        ms(base as f64),
                        ms(smoothed),
                    );
                }
            }
        }

        // A burst still draining long after it should be through
        if let (Some(drain), Some(last_arrival)) = (self.drain.as_ref(), self.last_arrival) {
            if drain.overdue(last_arrival, self.capacity) {
                drain.log_overrun(last_arrival, self.capacity, "burst still draining");
                signals.drain_overrun = true;
            }
        }

        signals.queue_delay = self
            .queue_delay_ns
            .map(|ns| Duration::from_nanos(ns.max(0.) as u64));
        signals.capacity = self.capacity;
        signals
    }

    fn measure_capacity(&mut self, drain: &Drain) {
        let arrival_spread = drain.last_arrival - drain.start_arrival;
        let send_spread = drain.last_send - drain.first_send;
        let bytes = drain.bytes - u64::from(drain.first_size);
        if bytes < MIN_CAPACITY_SAMPLE_BYTES || arrival_spread < MIN_CAPACITY_SAMPLE_NS {
            gst::trace!(
                CAT,
                "burst drained: {} bytes over {:.1} ms, too small to measure the capacity",
                drain.bytes,
                ms(arrival_spread as f64),
            );
            return;
        }

        let sample = bytes as f64 * 8. / arrival_spread as f64 * 1e9;
        let queue_limited = arrival_spread as f64 > QUEUE_LIMITED_SPREAD * send_spread as f64;
        let previous = self.capacity;
        self.capacity = Some(match self.capacity {
            // The bottleneck spread the burst out: that's its rate
            Some(capacity) if queue_limited => capacity + CAPACITY_ALPHA * (sample - capacity),
            // The pacer spread it out: the link can do at least that much
            Some(capacity) => capacity.max(sample),
            None => sample,
        });

        gst::debug!(
            CAT,
            "burst drained: {} bytes arrived over {:.1} ms, sent over {:.1} ms ({}), \
             {:.0} bit/s: capacity {:?} -> {:?}",
            drain.bytes,
            ms(arrival_spread as f64),
            ms(send_spread as f64),
            if queue_limited {
                "spread out by the bottleneck"
            } else {
                "spread out by the pacer, a lower bound"
            },
            sample,
            previous,
            self.capacity,
        );
    }

    /// Forget the queue state, e.g. after a decrease: the next samples
    /// should reflect the new bitrate
    pub fn reset_queue_delay(&mut self) {
        gst::trace!(CAT, "queuing delay reset");
        self.queue_delay_ns = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: i64 = 1_000_000;

    /// Simulates a FIFO bottleneck at `capacity` bit/sec with a fixed
    /// propagation delay, plus an extra standing queue
    struct Link {
        capacity: f64,
        free_at: i64,
        extra_ns: i64,
    }

    impl Link {
        fn send(&mut self, send_ns: i64, size: u32) -> i64 {
            let start = self.free_at.max(send_ns);
            self.free_at = start + (f64::from(size) * 8. / self.capacity * 1e9) as i64;
            self.free_at + 20 * MS + self.extra_ns
        }
    }

    fn packets(
        link: &mut Link,
        start_ns: i64,
        pace_bps: f64,
        n: u32,
        frame: (u32, u32),
        keyframe: bool,
    ) -> Vec<FeedbackPacket> {
        (0..n)
            .map(|i| {
                let send_ns = start_ns + (f64::from(i) * 1200. * 8. / pace_bps * 1e9) as i64;
                FeedbackPacket {
                    send_ns,
                    arrival_ns: Some(link.send(send_ns, 1200)),
                    size: 1200,
                    kind: PacketKind::Media,
                    keyframe,
                    large_frame: false,
                    frame: Some(frame),
                }
            })
            .collect()
    }

    /// 10 fps, 4 Mbps of delta frames, a keyframe 8x larger every
    /// `gop` frames (5.15 Mbps on average with a GOP of 25), on a link of
    /// `capacity`, unpaced (sent at 1 Gbps)
    fn run(capacity: f64, gop: u32, frames: u32, extra_ns: impl Fn(u32) -> i64) -> Vec<Signals> {
        let mut link = Link {
            capacity,
            free_at: 0,
            extra_ns: 0,
        };
        let mut detector = PacketDetector::default();
        (0..frames)
            .map(|i| {
                link.extra_ns = extra_ns(i);
                let keyframe = i % gop == 0;
                let n = if keyframe { 336 } else { 42 };
                let p = packets(&mut link, i64::from(i) * 100 * MS, 1e9, n, (1, i), keyframe);
                detector.update(&p)
            })
            .collect()
    }

    #[test]
    fn keyframes_alone_are_not_congestion() {
        // 400 kB keyframes take 400 ms through 8 Mbps, and with the delta
        // frames queued behind them the queue is ~800 ms deep
        let signals = run(8e6, 25, 100, |_| 0);
        assert!(signals.iter().all(|s| !s.drain_overrun));
        let last = signals.last().unwrap();
        assert!(
            last.queue_delay.unwrap() < Duration::from_millis(5),
            "{last:?}"
        );
        let capacity = last.capacity.unwrap();
        assert!((capacity - 8e6).abs() < 0.3e6, "{capacity}");
    }

    #[test]
    fn standing_queue_shows_in_delta_frames() {
        // Cross traffic builds a growing queue from frame 50 on
        let signals = run(20e6, 25, 80, |i| i64::from(i.saturating_sub(50)) * 5 * MS);
        assert!(signals[49].queue_delay.unwrap() < Duration::from_millis(2));
        assert!(signals[79].queue_delay.unwrap() > Duration::from_millis(80));
    }

    #[test]
    fn capacity_drop_during_a_keyframe_is_an_overrun() {
        let mut link = Link {
            capacity: 8e6,
            free_at: 0,
            extra_ns: 0,
        };
        let mut detector = PacketDetector::default();
        for i in 0..25 {
            let n = if i % 25 == 0 { 336 } else { 42 };
            detector.update(&packets(
                &mut link,
                i * 100 * MS,
                1e9,
                n,
                (1, i as u32),
                i % 25 == 0,
            ));
        }
        assert!(detector.capacity().is_some());

        // capacity drops to a third right as the next keyframe goes out
        link.capacity = 8e6 / 3.;
        let mut overrun = false;
        for i in 25..35 {
            let n = if i == 25 { 336 } else { 42 };
            overrun |= detector
                .update(&packets(
                    &mut link,
                    i * 100 * MS,
                    1e9,
                    n,
                    (1, i as u32),
                    i == 25,
                ))
                .drain_overrun;
        }
        assert!(overrun);
    }

    #[test]
    fn losses_are_split_by_kind() {
        let mut detector = PacketDetector::default();
        let mut p = FeedbackPacket {
            send_ns: 0,
            arrival_ns: None,
            size: 1200,
            kind: PacketKind::Media,
            keyframe: true,
            large_frame: false,
            frame: Some((1, 0)),
        };
        let mut batch = vec![p; 5];
        p.keyframe = false;
        p.frame = Some((1, 1));
        p.arrival_ns = Some(30 * MS);
        batch.extend([p; 10]);
        p.kind = PacketKind::RtxRepair;
        p.frame = None;
        p.arrival_ns = None;
        batch.push(p);

        let signals = detector.update(&batch);
        assert_eq!((signals.delta_packets, signals.delta_lost), (10, 0));
        assert_eq!((signals.burst_packets, signals.burst_lost), (6, 6));
        assert_eq!(signals.rtx_bits, 1200 * 8);
    }
}
