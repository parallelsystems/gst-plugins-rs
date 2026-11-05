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

/// Nanoseconds to milliseconds, for logs
fn ms(ns: impl Into<f64>) -> f64 {
    ns.into() / 1e6
}

/// Window over which the smallest one-way delay is taken as the delay of
/// an empty queue
const BASE_DELAY_WINDOW_NS: i64 = 10_000_000_000;
const BASE_DELAY_BUCKET_NS: i64 = 1_000_000_000;

/// Smoothing factor of the queuing delay, per delta frame
const QUEUE_DELAY_ALPHA: f64 = 0.25;

/// A packet whose one-way delay is this much above that of the burst's
/// first packet was queued behind the burst
const QUEUED_BEHIND_NS: i64 = 5_000_000;

/// The delay a burst builds up may be this many times what its bytes
/// explain at the estimated capacity...
const DRAIN_MARGIN: f64 = 1.5;
/// ... plus this, before the queue behind it counts as congestion
const DRAIN_SLACK_NS: f64 = 20_000_000.;
/// Without a capacity estimate, a burst building up more delay than this
/// is congestion
const MAX_DRAIN_NS: f64 = 2_000_000_000.;
/// Delta frames queued this much longer than the burst's own packets: the
/// queue grows on delta frames alone, sent faster than the link carries,
/// and the drain would never end
const DELTA_QUEUE_GROWTH_NS: i64 = 40_000_000;
/// A drain ends at the first delta frame packet sent this long after the
/// burst's last packet...
const DRAIN_MAX_TAIL_NS: i64 = 1_000_000_000;
/// ... or this long after the drain started, whatever the packets queued.
/// A drain whose first packet happened to arrive early compared to the
/// path's jitter sees every later packet as queued behind it, and with
/// bursts (keyframes, retransmissions) keeping it going it would otherwise
/// never end, leaving the queuing delay without samples.
const DRAIN_MAX_SPAN_NS: i64 = 2_000_000_000;

/// A drain spanning at least this many bytes and this long measures the
/// capacity
const MIN_CAPACITY_SAMPLE_BYTES: u64 = 10_000;
const MIN_CAPACITY_SAMPLE_NS: i64 = 2_000_000;
/// The drain was limited by the bottleneck, not by the pacer, when the
/// queue it built reached this fraction of how long it took to arrive...
const QUEUE_LIMITED_GROWTH: f64 = 0.2;
/// ... and at least this, above the receive-side jitter
const MIN_QUEUE_LIMITED_GROWTH_NS: i64 = 20_000_000;
/// The measured capacity is the median of this many recent samples from
/// drains the bottleneck spread out, so one jittered sample doesn't move it
const CAPACITY_SAMPLES: usize = 5;
/// A sample this far below the measured capacity replaces it outright: the
/// link got slower, and waiting for the median to follow would take a few
/// more bursts
const CAPACITY_DROP_RATIO: f64 = 0.8;
/// Without a new sample for this long, the measured capacity is stale
const CAPACITY_MAX_AGE_NS: i64 = 30_000_000_000;
/// A drain the pacer spread out only shows the link carried that rate at
/// that moment: it's a lower bound for this long
const LOWER_BOUND_MAX_AGE_NS: i64 = 10_000_000_000;
/// Packets that arrived over at least this long measure the delivery rate...
const DELIVERY_WINDOW_NS: i64 = 500_000_000;
/// ... which is the capacity if they all queued at least this much, so the
/// link never went idle...
const MIN_STANDING_QUEUE_NS: i64 = 20_000_000;
/// ... and the link spread them out over at least this many times as long as
/// they took to send. A path's delay going up queues every packet just the
/// same, but then they arrive as fast as they were sent.
const DELIVERY_SPREAD_RATIO: f64 = 1.15;

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

    /// Whether the packet is part of a regular delta frame
    fn is_delta(&self) -> bool {
        self.kind == PacketKind::Media && !self.is_burst()
    }

    /// Parses one `RTPPacerPacket`. None if it lacks a send time or size.
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
    /// Bottleneck capacity measured from bursts it spread out, bit/sec.
    /// None until measured, or once stale.
    pub capacity: Option<f64>,
    /// Rate the link recently carried at least, from bursts the pacer
    /// spread out, bit/sec
    pub capacity_lower_bound: Option<f64>,
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
    /// Send time of the last burst packet in the drain
    last_burst_send: i64,
    first_size: u32,
    bytes: u64,
    /// Most queue the drain built up, ns
    max_growth: i64,
    /// Most queue the burst's own packets saw, ns
    burst_max_growth: i64,
}

impl Drain {
    /// Starts a drain at the first packet of a burst
    fn new(p: &FeedbackPacket, arrival: i64) -> Self {
        Self {
            start_arrival: arrival,
            last_arrival: arrival,
            first_send: p.send_ns,
            last_send: p.send_ns,
            last_burst_send: p.send_ns,
            first_size: p.size,
            bytes: u64::from(p.size),
            max_growth: 0,
            burst_max_growth: 0,
        }
    }

    /// Adds a packet that arrived while the drain is ongoing
    fn add(&mut self, p: &FeedbackPacket, arrival: i64) {
        self.last_arrival = self.last_arrival.max(arrival);
        self.last_send = self.last_send.max(p.send_ns);
        self.bytes += u64::from(p.size);
        let growth = self.delay_growth(p.send_ns, arrival);
        self.max_growth = self.max_growth.max(growth);
        if p.is_burst() {
            self.burst_max_growth = self.burst_max_growth.max(growth);
            self.last_burst_send = self.last_burst_send.max(p.send_ns);
        }
    }

    /// Whether a packet sent at `send` is too long after the burst, or after
    /// the start of the drain, to still be queued behind it
    fn expired(&self, send: i64) -> bool {
        send - self.last_burst_send > DRAIN_MAX_TAIL_NS
            || send - self.first_send > DRAIN_MAX_SPAN_NS
    }

    /// How much more one-way delay a packet sent at `send` and arriving at
    /// `arrival` had than the burst's first packet: the queue the burst
    /// built up. Zero if the pacer sent it no faster than the link carries.
    fn delay_growth(&self, send: i64, arrival: i64) -> i64 {
        (arrival - self.start_arrival) - (send - self.first_send)
    }

    /// Whether a packet sent at `send` and arriving at `arrival` saw more
    /// queue than the burst explains
    fn overdue(&self, send: i64, arrival: i64, capacity: Option<f64>) -> bool {
        let growth = self.delay_growth(send, arrival) as f64;
        match self.expected_growth_ns(send, capacity) {
            Some(expected) => growth > expected * DRAIN_MARGIN + DRAIN_SLACK_NS,
            None => growth > MAX_DRAIN_NS,
        }
    }

    /// Whether a delta frame packet sent at `send` and arriving at `arrival`
    /// queued well beyond what the burst's packets did. Behind a burst sent
    /// above a link that keeps up with the delta frames, the queue peaks with
    /// the burst's last packet and shrinks from there.
    fn queue_growing(&self, send: i64, arrival: i64) -> bool {
        self.delay_growth(send, arrival) > self.burst_max_growth + DELTA_QUEUE_GROWTH_NS
    }

    /// The queue left at `send` by sending the drain's bytes faster than
    /// `capacity`: what they need at `capacity` minus how long they took
    /// to send
    fn expected_growth_ns(&self, send: i64, capacity: Option<f64>) -> Option<f64> {
        capacity.map(|capacity| {
            (self.bytes as f64 * 8. / capacity * 1e9 - (send - self.first_send) as f64).max(0.)
        })
    }

    /// Logs the delay the drain built up against what it should have
    fn log_overrun(&self, send: i64, arrival: i64, capacity: Option<f64>, what: &str) {
        gst::debug!(
            CAT,
            "{what}: {:.1} ms more delay ({:.1} ms for the burst) after draining for {:.1} ms, \
             its {} bytes sent over {:.1} ms explain {:?} ms at capacity {:?}",
            ms(self.delay_growth(send, arrival) as f64),
            ms(self.burst_max_growth as f64),
            ms((arrival - self.start_arrival) as f64),
            self.bytes,
            ms((send - self.first_send) as f64),
            self.expected_growth_ns(send, capacity).map(ms),
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
    /// Adds a one-way delay sample and ages out buckets older than the window
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

    /// Smallest one-way delay in the window
    fn get(&self) -> Option<i64> {
        self.buckets.iter().map(|(_, min)| *min).min()
    }
}

/// A packet that arrived, for the delivery rate
#[derive(Debug, Clone, Copy)]
struct Delivered {
    send: i64,
    arrival: i64,
    size: u32,
    /// Its one-way delay over the base delay, ns
    queue: i64,
}

/// Packets that arrived over the last DELIVERY_WINDOW_NS, in feedback order
#[derive(Debug, Default)]
struct DeliveryWindow {
    packets: VecDeque<Delivered>,
}

impl DeliveryWindow {
    /// Adds a packet, ageing out those not needed to span a window before it
    fn push(&mut self, packet: Delivered) {
        self.packets.push_back(packet);
        while self
            .packets
            .get(1)
            .is_some_and(|p| packet.arrival - p.arrival >= DELIVERY_WINDOW_NS)
        {
            self.packets.pop_front();
        }
    }

    /// The rate the window's packets arrived at, bit/sec, if it is the
    /// capacity: they arrived over a whole window, all queued, and the link
    /// spread them out. Starts a new window when it is.
    fn take_capacity(&mut self) -> Option<f64> {
        let first_arrival = self.packets.iter().map(|p| p.arrival).min()?;
        let last_arrival = self.packets.iter().map(|p| p.arrival).max()?;
        let first_send = self.packets.iter().map(|p| p.send).min()?;
        let last_send = self.packets.iter().map(|p| p.send).max()?;
        let arrival_spread = last_arrival - first_arrival;
        let send_spread = last_send - first_send;
        let min_queue = self.packets.iter().map(|p| p.queue).min()?;
        if arrival_spread < DELIVERY_WINDOW_NS
            || min_queue < MIN_STANDING_QUEUE_NS
            || (arrival_spread as f64) < DELIVERY_SPREAD_RATIO * send_spread as f64
        {
            return None;
        }

        // The first packet's bytes went out before the window started
        let bytes: u64 = self.packets.iter().skip(1).map(|p| u64::from(p.size)).sum();
        let rate = bytes as f64 * 8. / arrival_spread as f64 * 1e9;
        gst::debug!(
            CAT,
            "delivery rate under a standing queue: {} packets, {} bytes arrived over {:.1} ms, \
             sent over {:.1} ms, queuing at least {:.1} ms: {:.0} bit/s",
            self.packets.len(),
            bytes,
            ms(arrival_spread as f64),
            ms(send_spread as f64),
            ms(min_queue as f64),
            rate,
        );
        self.packets.clear();
        Some(rate)
    }
}

/// The bottleneck's rate, measured from drains it spread out, kept apart
/// from what drains the pacer spread out show the link can at least carry.
/// Times are send times, in ns.
#[derive(Debug, Default)]
struct Capacity {
    /// Recent bottleneck-limited samples, bit/sec, oldest first
    samples: VecDeque<f64>,
    /// When the last of them was taken
    measured_at: Option<i64>,
    /// Highest recent pacer-limited sample, bit/sec, and when it was taken
    lower_bound: Option<(f64, i64)>,
}

impl Capacity {
    /// Adds a sample from a drain the bottleneck spread out. Returns whether
    /// it replaced the estimate, being well below it or the estimate stale.
    fn add_measurement(&mut self, sample: f64, at: i64) -> bool {
        let replace = self
            .measured(at)
            .map_or(true, |capacity| sample < CAPACITY_DROP_RATIO * capacity);
        if replace {
            self.samples.clear();
        }
        self.samples.push_back(sample);
        while self.samples.len() > CAPACITY_SAMPLES {
            self.samples.pop_front();
        }
        self.measured_at = Some(at);
        replace
    }

    /// Adds a sample from a drain the pacer spread out
    fn add_lower_bound(&mut self, sample: f64, at: i64) {
        if self.lower_bound(at).map_or(true, |bound| sample >= bound) {
            self.lower_bound = Some((sample, at));
        }
    }

    /// Median of the recent samples, None if there are none or they're stale
    fn measured(&self, now: i64) -> Option<f64> {
        if now - self.measured_at? > CAPACITY_MAX_AGE_NS {
            return None;
        }
        let mut sorted: Vec<f64> = self.samples.iter().copied().collect();
        sorted.sort_by(f64::total_cmp);
        let mid = sorted.len() / 2;
        Some(if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.
        } else {
            sorted[mid]
        })
    }

    /// The lower bound, None once it has aged out
    fn lower_bound(&self, now: i64) -> Option<f64> {
        self.lower_bound
            .filter(|(_, at)| now - at <= LOWER_BOUND_MAX_AGE_NS)
            .map(|(bound, _)| bound)
    }

    /// What drains are judged against: the measured capacity, or failing
    /// that the lower bound, which only makes the judgement more lenient
    fn reference(&self, now: i64) -> Option<f64> {
        self.measured(now).or_else(|| self.lower_bound(now))
    }
}

#[derive(Debug, Default)]
pub(crate) struct PacketDetector {
    base_delay: BaseDelay,
    capacity: Capacity,
    /// Latest send time seen, ns
    now_ns: i64,
    drain: Option<Drain>,
    /// Recent packets, to measure the capacity from while a queue stands
    delivered: DeliveryWindow,
    last_delta_frame: Option<(u32, u32)>,
    queue_delay_ns: Option<f64>,
}

impl PacketDetector {
    /// Measured bottleneck capacity, in bit/sec
    #[cfg(test)]
    pub fn capacity(&self) -> Option<f64> {
        self.capacity.measured(self.now_ns)
    }

    /// Rate the link recently carried at least, in bit/sec
    #[cfg(test)]
    pub fn capacity_lower_bound(&self) -> Option<f64> {
        self.capacity.lower_bound(self.now_ns)
    }

    /// Processes the packets of one feedback, in TWCC order
    pub fn update(&mut self, packets: &[FeedbackPacket]) -> Signals {
        let mut signals = Signals::default();

        if let (Some(first), Some(last)) = (
            packets.iter().map(|p| p.send_ns).min(),
            packets.iter().map(|p| p.send_ns).max(),
        ) {
            signals.send_span_ns = last - first;
            self.now_ns = self.now_ns.max(last);
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
            if let Some(base) = self.base_delay.get() {
                self.delivered.push(Delivered {
                    send: p.send_ns,
                    arrival,
                    size: p.size,
                    queue: owd - base,
                });
            }

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
                // A paced burst on a link that keeps up with it leaves no
                // queue: the delta frames after it end the drain right away
                let queued_behind = drain.delay_growth(p.send_ns, arrival) > QUEUED_BEHIND_NS;
                let capacity = self.capacity.reference(p.send_ns);
                let overdue = drain.overdue(p.send_ns, arrival, capacity);
                let growing = drain.queue_growing(p.send_ns, arrival);

                let overrun = queued_behind && (overdue || growing);

                if !overrun && drain.expired(p.send_ns) {
                    // How long it ran says nothing about the link's rate,
                    // and this packet's queuing delay is the queue's
                    gst::debug!(
                        CAT,
                        "drain ended {:.1} ms after its last burst packet, {:.1} ms after it \
                         started, with {:.1} ms more delay than its first packet",
                        ms((p.send_ns - drain.last_burst_send) as f64),
                        ms((p.send_ns - drain.first_send) as f64),
                        ms(drain.delay_growth(p.send_ns, arrival) as f64),
                    );
                    self.drain = None;
                } else if queued_behind && !overrun {
                    // Queued behind the burst: says nothing about the network
                    drain.add(p, arrival);
                    continue;
                } else {
                    if overrun {
                        drain.log_overrun(
                            p.send_ns,
                            arrival,
                            capacity,
                            if growing {
                                "delta frames queued longer than the burst before them"
                            } else {
                                "delta frames still queued behind a burst"
                            },
                        );
                        signals.drain_overrun = true;
                    }
                    let drain = self.drain.take().unwrap();
                    self.measure_capacity(&drain);
                }
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
        if let Some(drain) = self.drain.as_ref() {
            let capacity = self.capacity.reference(drain.last_send);
            if drain.overdue(drain.last_send, drain.last_arrival, capacity) {
                drain.log_overrun(
                    drain.last_send,
                    drain.last_arrival,
                    capacity,
                    "burst still draining",
                );
                signals.drain_overrun = true;
            }
        }

        if let Some(rate) = self.delivered.take_capacity() {
            let previous = self.capacity.measured(self.now_ns);
            let replaced = self.capacity.add_measurement(rate, self.now_ns);
            gst::debug!(
                CAT,
                "capacity {:?} -> {:?}{} from the delivery rate",
                previous,
                self.capacity.measured(self.now_ns),
                if replaced && previous.is_some() {
                    ", replaced"
                } else {
                    ""
                },
            );
        }

        signals.queue_delay = self
            .queue_delay_ns
            .map(|ns| Duration::from_nanos(ns.max(0.) as u64));
        signals.capacity = self.capacity.measured(self.now_ns);
        signals.capacity_lower_bound = self.capacity.lower_bound(self.now_ns);
        signals
    }

    /// Estimates capacity from how a finished drain spread out on arrival.
    /// If the bottleneck spread it (it queued there) the sample measures the
    /// capacity; if the pacer did, it's only a lower bound.
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

        // Delta frames queued behind a burst stretch how long the drain took
        // to send as much as to arrive: whether it queued tells better
        let queue_limited = drain.max_growth
            > MIN_QUEUE_LIMITED_GROWTH_NS
                .max((QUEUE_LIMITED_GROWTH * arrival_spread as f64) as i64);
        // Packets the pacer sent in one go can arrive closer together than
        // the link's sustained rate: for a lower bound, go by the slower of
        // sending and arriving
        let spread = if queue_limited {
            arrival_spread
        } else {
            arrival_spread.max(send_spread)
        };
        let sample = bytes as f64 * 8. / spread as f64 * 1e9;
        let at = drain.last_send;
        let previous = self.capacity.measured(at);
        let replaced = if queue_limited {
            // The bottleneck spread the burst out: that's its rate
            self.capacity.add_measurement(sample, at)
        } else {
            // The pacer spread it out: the link can do at least that much
            self.capacity.add_lower_bound(sample, at);
            false
        };

        gst::debug!(
            CAT,
            "burst drained: {} bytes arrived over {:.1} ms, sent over {:.1} ms ({}), \
             {:.0} bit/s: capacity {:?} -> {:?}{} (lower bound {:?})",
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
            self.capacity.measured(at),
            if replaced && previous.is_some() {
                ", replaced"
            } else {
                ""
            },
            self.capacity.lower_bound(at),
        );
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
        /// Arrival time of a packet sent at `send_ns`
        fn send(&mut self, send_ns: i64, size: u32) -> i64 {
            let start = self.free_at.max(send_ns);
            self.free_at = start + (f64::from(size) * 8. / self.capacity * 1e9) as i64;
            self.free_at + 20 * MS + self.extra_ns
        }
    }

    /// `n` packets of one frame, sent at `pace_bps` through `link`
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
    fn paced_keyframes_on_a_fast_link_are_not_congestion() {
        // What the Jetson did on an idle LAN: 10 fps, keyframes every 25
        // frames, and frames sized like one in between. The pacer sends a
        // few packets per 5 ms tick, faster at first, then spreads bursts
        // over hundreds of ms once the bitrate is lower.
        let mut link = Link {
            capacity: 1e9,
            free_at: 0,
            extra_ns: 0,
        };
        let mut detector = PacketDetector::default();
        // The pacer's next tick: frames wait behind the burst before them
        let mut tick = 0;
        for i in 0..250u32 {
            tick = tick.max(i64::from(i) * 100 * MS);
            let (n, burst) = match i % 25 {
                0 => (110, true),
                12 => (80, true),
                _ => (20, false),
            };
            // 15 Mbps catching up at first, then 1.9 Mbps
            let per_tick = match (burst, i < 50) {
                (true, true) => 8,
                _ => 1,
            };
            let p: Vec<_> = (0..n)
                .map(|j| {
                    let send_ns = tick;
                    if (j + 1) % per_tick == 0 || j + 1 == n {
                        tick += 5 * MS;
                    }
                    FeedbackPacket {
                        send_ns,
                        arrival_ns: Some(link.send(send_ns, 1200)),
                        size: 1200,
                        kind: PacketKind::Media,
                        keyframe: burst,
                        large_frame: false,
                        frame: Some((1, i)),
                    }
                })
                .collect();
            let signals = detector.update(&p);
            assert!(!signals.drain_overrun, "overrun at frame {i}");
        }
        // The link never limited a burst: nothing measured, only a bound
        assert_eq!(detector.capacity(), None);
        assert!(detector.capacity_lower_bound().is_some());
    }

    #[test]
    fn capacity_is_the_median_and_follows_drops_at_once() {
        let mut capacity = Capacity::default();
        for (i, sample) in [8e6, 7.8e6, 8.2e6, 8e6].into_iter().enumerate() {
            capacity.add_measurement(sample, i as i64 * 1000 * MS);
        }
        // one jittered sample doesn't move it
        assert!(!capacity.add_measurement(20e6, 4000 * MS));
        assert_eq!(capacity.measured(4000 * MS), Some(8e6));
        // the link getting slower does, right away
        assert!(capacity.add_measurement(2e6, 5000 * MS));
        assert_eq!(capacity.measured(5000 * MS), Some(2e6));
        // and it goes stale without new samples
        assert_eq!(capacity.measured(5000 * MS + CAPACITY_MAX_AGE_NS + 1), None);
    }

    #[test]
    fn lower_bound_ages_out_and_never_becomes_the_capacity() {
        let mut capacity = Capacity::default();
        // a burst sent and arriving at 27 Mbps, as at 3:34 in ladder_minimal
        capacity.add_lower_bound(27e6, 0);
        capacity.add_lower_bound(1e6, 1000 * MS);
        assert_eq!(capacity.measured(1000 * MS), None);
        assert_eq!(capacity.lower_bound(1000 * MS), Some(27e6));
        assert_eq!(capacity.reference(1000 * MS), Some(27e6));
        assert_eq!(capacity.lower_bound(LOWER_BOUND_MAX_AGE_NS + 1), None);
        // once measured, the measurement is what drains are judged against
        capacity.add_measurement(1.8e6, 2000 * MS);
        assert_eq!(capacity.reference(2000 * MS), Some(1.8e6));
    }

    #[test]
    fn capacity_drop_is_measured_with_the_next_burst() {
        let mut link = Link {
            capacity: 8e6,
            free_at: 0,
            extra_ns: 0,
        };
        let mut detector = PacketDetector::default();
        for i in 0..50 {
            let keyframe = i % 25 == 0;
            let n = if keyframe { 336 } else { 42 };
            detector.update(&packets(
                &mut link,
                i * 100 * MS,
                1e9,
                n,
                (1, i as u32),
                keyframe,
            ));
        }
        let before = detector.capacity().unwrap();
        assert!((before - 8e6).abs() < 0.3e6, "{before}");

        // the link drops to 2 Mbps: one keyframe, then a delta frame once
        // it has drained
        link.capacity = 2e6;
        detector.update(&packets(&mut link, 5000 * MS, 1e9, 336, (1, 50), true));
        detector.update(&packets(&mut link, 7000 * MS, 1e9, 5, (1, 51), false));
        let after = detector.capacity().unwrap();
        assert!((after - 2e6).abs() < 0.1e6, "{after}");
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
    fn delta_frames_above_capacity_end_the_drain() {
        // As in ladder_degraded: capacity measured at 2 Mbps, then the
        // target climbs past it while a keyframe drains
        let mut link = Link {
            capacity: 2e6,
            free_at: 0,
            extra_ns: 0,
        };
        let mut detector = PacketDetector::default();
        for i in 0..25 {
            let keyframe = i % 25 == 0;
            let n = if keyframe { 50 } else { 10 };
            let signals = detector.update(&packets(
                &mut link,
                i * 100 * MS,
                1e9,
                n,
                (1, i as u32),
                keyframe,
            ));
            assert!(!signals.drain_overrun, "overrun at frame {i}");
        }
        assert!(detector.capacity().is_some());

        // a keyframe, then delta frames at 3 Mbps: 50% above the link
        let mut overrun_at = None;
        for i in 25..50 {
            let keyframe = i == 25;
            let n = if keyframe { 50 } else { 31 };
            let signals = detector.update(&packets(
                &mut link,
                i * 100 * MS,
                1e9,
                n,
                (1, i as u32),
                keyframe,
            ));
            if signals.drain_overrun {
                overrun_at.get_or_insert(i);
            }
        }
        // the queue grows 50 ms per 100 ms frame: seen within a few frames,
        // and delta frames show it as queuing delay from then on
        let overrun_at = overrun_at.expect("no overrun");
        assert!(overrun_at <= 30, "overrun only at frame {overrun_at}");
        let queue_delay = detector
            .update(&packets(&mut link, 5000 * MS, 1e9, 31, (1, 50), false))
            .queue_delay
            .unwrap();
        assert!(queue_delay > Duration::from_millis(100), "{queue_delay:?}");
    }

    /// Delta frames only, `n` packets every 100 ms, unpaced, through `link`
    #[test]
    fn early_burst_packet_does_not_keep_the_drain_going() {
        // As in hitl1105: each keyframe's first packet arrives 10 ms early
        // (jitter), so every packet after it looks queued behind it, and
        // the next keyframe comes before the drain is over. Without a limit
        // the drain never ends and the queuing delay is never sampled.
        let mut link = Link {
            capacity: 100e6,
            free_at: 0,
            extra_ns: 0,
        };
        let mut detector = PacketDetector::default();
        let mut samples = 0;
        for i in 0..100u32 {
            let keyframe = i % 10 == 0;
            let n = if keyframe { 50 } else { 10 };
            let mut p = packets(&mut link, i64::from(i) * 100 * MS, 1e9, n, (1, i), keyframe);
            if keyframe {
                *p[0].arrival_ns.as_mut().unwrap() -= 10 * MS;
            }
            let signals = detector.update(&p);
            assert!(!signals.drain_overrun, "overrun at frame {i}");
            if i >= 50 {
                samples += signals.delay_samples;
            }
        }
        // a sample at least every DRAIN_MAX_SPAN_NS
        assert!(samples >= 2, "{samples} delay samples");
        let queue_delay = detector.queue_delay_ns.unwrap();
        assert!(queue_delay < 15. * MS as f64, "{queue_delay}");
    }

    fn delta_frames(
        link: &mut Link,
        detector: &mut PacketDetector,
        frames: std::ops::Range<u32>,
        n: u32,
        extra_ns: impl Fn(u32) -> i64,
    ) {
        for i in frames {
            link.extra_ns = extra_ns(i);
            detector.update(&packets(
                link,
                i64::from(i) * 100 * MS,
                1e9,
                n,
                (1, i),
                false,
            ));
        }
    }

    #[test]
    fn standing_queue_measures_the_capacity() {
        // The link drops from 20 to 1 Mbps under 2 Mbps of delta frames,
        // with no keyframe to measure it with
        let mut link = Link {
            capacity: 20e6,
            free_at: 0,
            extra_ns: 0,
        };
        let mut detector = PacketDetector::default();
        delta_frames(&mut link, &mut detector, 0..20, 21, |_| 0);
        assert_eq!(detector.capacity(), None);

        link.capacity = 1e6;
        delta_frames(&mut link, &mut detector, 20..30, 21, |_| 0);
        let capacity = detector.capacity().expect("not measured");
        assert!((capacity - 1e6).abs() < 0.1e6, "{capacity}");
    }

    #[test]
    fn longer_path_is_not_a_capacity() {
        // The path's delay goes up by 60 ms for a while, as in no-netem1:
        // every packet queues, but the link still keeps up
        let mut link = Link {
            capacity: 20e6,
            free_at: 0,
            extra_ns: 0,
        };
        let mut detector = PacketDetector::default();
        delta_frames(&mut link, &mut detector, 0..100, 21, |i| {
            if (20..80).contains(&i) {
                60 * MS
            } else {
                0
            }
        });
        assert_eq!(detector.capacity(), None);
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
