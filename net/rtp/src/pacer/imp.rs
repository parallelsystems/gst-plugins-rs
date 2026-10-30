// SPDX-License-Identifier: MPL-2.0

/**
 * element-rtppacer:
 *
 * Paces outgoing RTP packets so they leave at a steady rate instead of in
 * the bursts the encoders produce, and reports, packet by packet, what the
 * transport-wide congestion control feedback says happened to them.
 *
 * Like `rtpgccbwe`, this element should be placed right before an
 * `rtpsession` (e.g. as the `webrtcbin` aux sender, which puts it after
 * `rtprtxsend`) and only reports feedback when TWCC is enabled.
 *
 * Pacing: packets leave at #rtppacer:pacing-factor times #rtppacer:bitrate.
 * The queue holds at most one frame interval (and at least 30 ms) of data
 * at that rate, or a few frame intervals while a keyframe, or a frame much
 * larger than the usual delta frame, is queued. Rather than sending what
 * doesn't fit as a burst, the pacer then speeds up just enough to drain the
 * queue within that time. Retransmissions skip ahead of queued media.
 *
 * Feedback: as packets go through, the pacer classifies them (media in a
 * keyframe or a delta frame, retransmission answering a NACK, other RTX),
 * then matches the `RTPTWCCPackets` events `rtpsession` sends upstream
 * against what it sent, and emits #rtppacer::packet-feedback with the
 * annotated packets.
 *
 * Keyframes are recognised from the buffers' DELTA_UNIT flag, which the
 * H.264/H.265 payloaders copy from the encoder. Payloaders that never set
 * the flag get none of their frames treated as keyframes, only as large
 * frames when they are.
 */
use gst::{glib, prelude::*, subclass::prelude::*};
use once_cell::sync::Lazy;
use std::{
    collections::VecDeque,
    sync::Mutex,
    time::{Duration, Instant},
};

static CAT: Lazy<gst::DebugCategory> = Lazy::new(|| {
    gst::DebugCategory::new(
        "rtppacer",
        gst::DebugColorFlags::empty(),
        Some("RTP pacer with per-packet TWCC feedback"),
    )
});

const DEFAULT_BITRATE: u32 = 2_048_000;
const DEFAULT_PACING_FACTOR: f64 = 1.5;
const DEFAULT_FRAMERATE: f64 = 10.;
const DEFAULT_GOP_SIZE: u32 = 25;

/// How often the pacer wakes up to send what its budget allows
const TICK: Duration = Duration::from_millis(5);

/// The queue holds at least this much data at the pacing rate before the
/// pacer speeds up
const MIN_PACING_QUEUE: Duration = Duration::from_millis(30);

/// While a keyframe is queued the queue may hold this many frame intervals
/// of data at the pacing rate...
const KEYFRAME_QUEUE_FRAMES: f64 = 3.;

/// ... but no more than this fraction of a GOP, so a keyframe is through
/// well before the next one is due
const KEYFRAME_QUEUE_GOP_FRACTION: f64 = 0.5;

/// A frame this many times the average delta frame is paced like a keyframe,
/// e.g. an intra frame the encoder doesn't flag as one
const LARGE_FRAME_RATIO: f64 = 3.;

/// Smoothing factor of the average delta frame size
const DELTA_FRAME_SIZE_ALPHA: f64 = 0.1;

/// Sent packets and frames are kept this long waiting for their feedback
const SENT_LOG_MAX_AGE: Duration = Duration::from_secs(5);

/// A retransmission request is answered within this long; RTX packets that
/// don't answer one are padding
const RTX_REQUEST_MAX_AGE: Duration = Duration::from_secs(1);

/// What a packet is, as far as feedback is concerned
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Media packet, from the payloader
    Media,
    /// Retransmission answering a NACK
    RtxRepair,
    /// Retransmission nobody asked for, sent as bandwidth probing padding
    RtxPadding,
}

impl Kind {
    /// Name used for the `kind` field of the feedback
    fn as_str(self) -> &'static str {
        match self {
            Kind::Media => "media",
            Kind::RtxRepair => "rtx-repair",
            Kind::RtxPadding => "rtx-padding",
        }
    }
}

/// What the pacer knows about a packet going through
#[derive(Debug, Clone, Copy)]
struct PacketInfo {
    /// RTP payload type
    pt: u8,
    /// RTP SSRC
    ssrc: u32,
    /// RTP sequence number
    seqnum: u16,
    /// RTP timestamp
    rtp_ts: u32,
    /// RTP marker bit, set on the last packet of a frame
    marker: bool,
    /// RTP payload size, what TWCC feedback reports as the size
    payload_len: u32,
    /// rtpsession only gives packets with a header extension a TWCC seqnum
    has_extension: bool,
    /// Media or RTX
    kind: Kind,
    /// Media: the frame it belongs to (Frame::uid)
    frame: Option<u64>,
    /// Media: whether it's the frame's first packet
    frame_start: bool,
    /// RTX: the sequence number of the packet it carries
    osn: Option<u16>,
}

/// A video frame, as seen in its packets
#[derive(Debug, Clone)]
struct Frame {
    /// Unique id, increasing across streams
    uid: u64,
    /// RTP timestamp shared by its packets
    rtp_ts: u32,
    /// Payload bytes so far
    bytes: u64,
    /// Packets so far
    packets: u32,
    /// Its packets aren't flagged as delta units
    keyframe: bool,
    /// Keyframe that came before the GOP was over, i.e. on request (PLI/FIR)
    requested: bool,
    /// Not flagged as a keyframe, but sized like one
    large: bool,
    /// Its last packet (marker bit) has been seen
    complete: bool,
    /// When its first packet arrived
    created: Instant,
}

impl Frame {
    /// Whether the frame is sent as a burst the queue can't absorb in one
    /// frame interval: a keyframe or a large delta frame
    fn is_burst(&self) -> bool {
        self.keyframe || self.large
    }
}

/// Per-SSRC frame tracking
#[derive(Debug)]
struct MediaStream {
    /// RTP SSRC of the stream
    ssrc: u32,
    /// The payloader flags delta units, so a buffer without the flag
    /// belongs to a keyframe
    delta_flags_seen: bool,
    /// Frames since the last keyframe, None before the first one
    frames_since_keyframe: Option<u64>,
    /// Moving average of the delta frame size, to spot large frames
    avg_delta_frame_bytes: Option<f64>,
    /// The frame being received
    current: Option<u64>,
}

/// A packet waiting in the pacer
struct Queued {
    /// The RTP packet
    buffer: gst::Buffer,
    /// What's known about it, None if it isn't RTP
    info: Option<PacketInfo>,
    /// When it entered the queue
    enqueued: Instant,
    /// When it must have left the pacer, set when it's queued
    deadline: Instant,
}

/// What the pacer pushes downstream
enum Output {
    /// An RTP packet
    Buffer(gst::Buffer),
    /// A serialized event that was queued behind packets
    Event(gst::Event),
}

/// A packet that left the pacer with a TWCC extension
#[derive(Debug)]
struct Sent {
    /// Position in the order rtpsession numbers packets
    index: u64,
    /// What the pacer knew about it
    info: PacketInfo,
    /// When it left the pacer
    at: Instant,
}

/// One packet of an `RTPTWCCPackets` event
#[derive(Debug, Clone, Copy)]
struct Feedback {
    /// Extended TWCC seqnum
    seqnum: i64,
    /// Send time on rtpsession's clock
    local_ts: u64,
    /// Arrival time on the receiver's clock, None if lost
    remote_ts: Option<u64>,
    /// Payload type rtpsession saw, to check alignment
    pt: u8,
    /// RTP payload size, to check alignment
    size: u32,
    /// Reported lost by the receiver
    lost: bool,
}

impl Feedback {
    /// Parses one packet of an `RTPTWCCPackets` event, with its TWCC seqnum.
    /// None for packets rtpsession has no send time for.
    fn from_structure(s: &gst::StructureRef) -> Option<(u16, Self)> {
        // rtpsession leaves these at NONE for packets it lost track of
        let local_ts = s.get::<u64>("local-ts").ok().filter(|ts| *ts != u64::MAX)?;
        let seqnum = s.get::<u32>("seqnum").ok()? as u16;
        let lost = s.get::<bool>("lost").unwrap_or(false);
        let remote_ts = s
            .get::<u64>("remote-ts")
            .ok()
            .filter(|ts| !lost && *ts != u64::MAX);

        Some((
            seqnum,
            Feedback {
                seqnum: 0,
                local_ts,
                remote_ts,
                pt: s.get::<u8>("payload-type").unwrap_or(0),
                size: s.get::<u32>("size").ok()?,
                lost,
            },
        ))
    }
}

struct State {
    /// Target bitrate from the congestion controller, bit/sec
    bitrate: u32,
    /// Pacing rate as a multiple of the bitrate
    pacing_factor: f64,
    /// Encoder frame rate, sets the queue times
    framerate: f64,
    /// Encoder keyframe interval in frames, 0 if unknown
    gop_size: u32,
    /// rtprtxsend's payload-type-map, as set
    rtx_pt_map: Option<gst::Structure>,
    /// RTX payload types from rtx_pt_map
    rtx_pts: Vec<u8>,

    /// RTX packets, sent before any media
    rtx_queue: VecDeque<Queued>,
    /// Media packets and anything that isn't RTX
    media_queue: VecDeque<Queued>,
    /// Serialized events, each with the number of media packets queued
    /// before it, so they go out in order with the media
    events: VecDeque<(u64, gst::Event)>,
    /// Media packets queued since start, to place events
    media_enqueued: u64,
    /// Media packets sent since start, to release events
    media_dequeued: u64,
    /// Bytes in both queues
    queued_bytes: u64,
    /// Bits sent ahead of (negative) or behind (positive) the budget
    budget_offset: f64,
    /// When the budget was last updated
    last_push: Option<Instant>,
    /// Sending faster than the pacing rate to keep within the queue time
    speeding_up: bool,

    /// Media streams by SSRC
    streams: Vec<MediaStream>,
    /// Ordered by uid
    frames: VecDeque<Frame>,
    /// uid for the next new frame
    next_frame_uid: u64,
    /// Sequence numbers rtpsession asked rtprtxsend to retransmit
    rtx_requests: VecDeque<(u16, Instant)>,

    /// Packets that got a TWCC seqnum, in the order rtpsession numbered them
    sent: VecDeque<Sent>,
    /// Index for the next packet added to `sent`
    next_index: u64,
    /// Extended TWCC seqnum minus index in `sent`, once lined up
    offset: Option<i64>,
    /// Highest extended TWCC seqnum seen in feedback
    twcc_extended_seqnum: Option<i64>,

    /// Clock entry the srcpad task waits on between ticks
    clock_entry: Option<gst::SingleShotClockId>,
    /// Last flow return from downstream, returned by the chain function
    flow_return: Result<gst::FlowSuccess, gst::FlowError>,
}

impl Default for State {
    fn default() -> Self {
        Self {
            bitrate: DEFAULT_BITRATE,
            pacing_factor: DEFAULT_PACING_FACTOR,
            framerate: DEFAULT_FRAMERATE,
            gop_size: DEFAULT_GOP_SIZE,
            rtx_pt_map: None,
            rtx_pts: Vec::new(),
            rtx_queue: VecDeque::new(),
            media_queue: VecDeque::new(),
            events: VecDeque::new(),
            media_enqueued: 0,
            media_dequeued: 0,
            queued_bytes: 0,
            budget_offset: 0.,
            last_push: None,
            speeding_up: false,
            streams: Vec::new(),
            frames: VecDeque::new(),
            next_frame_uid: 0,
            rtx_requests: VecDeque::new(),
            sent: VecDeque::new(),
            next_index: 0,
            offset: None,
            twcc_extended_seqnum: None,
            clock_entry: None,
            flow_return: Err(gst::FlowError::Flushing),
        }
    }
}

impl State {
    /// Rate in bit/sec the queue drains at when it's within its queue time
    fn pacing_rate(&self) -> f64 {
        self.bitrate as f64 * self.pacing_factor
    }

    /// How long the queue may take to drain at the pacing rate with only
    /// delta frames in it: one frame interval, at least MIN_PACING_QUEUE
    fn frame_queue_time(&self) -> Duration {
        Duration::from_secs_f64(1. / self.framerate).max(MIN_PACING_QUEUE)
    }

    /// Same, while a keyframe or large frame is queued
    fn keyframe_queue_time(&self) -> Duration {
        let mut secs = KEYFRAME_QUEUE_FRAMES / self.framerate;
        if self.gop_size > 0 {
            secs = secs.min(KEYFRAME_QUEUE_GOP_FRACTION * self.gop_size as f64 / self.framerate);
        }
        Duration::from_secs_f64(secs).max(self.frame_queue_time())
    }

    /// Takes rtprtxsend's `payload-type-map` (media PT -> RTX PT) to tell
    /// RTX packets apart
    fn set_rtx_pt_map(&mut self, map: Option<gst::Structure>) {
        self.rtx_pts = map
            .iter()
            .flat_map(|s| s.iter())
            .filter_map(|(_, value)| {
                value
                    .get::<u32>()
                    .ok()
                    .or_else(|| value.get::<i32>().ok().and_then(|v| u32::try_from(v).ok()))
                    .and_then(|pt| u8::try_from(pt).ok())
            })
            .collect();
        self.rtx_pt_map = map;
    }

    /// Frame by uid, if it's still tracked
    fn frame(&self, uid: u64) -> Option<&Frame> {
        self.frames
            .binary_search_by_key(&uid, |f| f.uid)
            .ok()
            .map(|i| &self.frames[i])
    }

    /// Mutable frame by uid, if it's still tracked
    fn frame_mut(&mut self, uid: u64) -> Option<&mut Frame> {
        self.frames
            .binary_search_by_key(&uid, |f| f.uid)
            .ok()
            .map(|i| &mut self.frames[i])
    }

    /// Reads what the pacer needs from an RTP packet: RTX packets are
    /// repairs if a retransmission was requested for their original sequence number (OSN), padding
    /// otherwise; media packets are added to their frame. None if it isn't RTP.
    fn classify(
        &mut self,
        element: &super::Pacer,
        buffer: &gst::BufferRef,
        now: Instant,
    ) -> Option<PacketInfo> {
        let rtp = gst_rtp::RTPBuffer::from_buffer_readable(buffer).ok()?;
        let pt = rtp.payload_type();
        let ssrc = rtp.ssrc();
        let seqnum = rtp.seq();
        let rtp_ts = rtp.timestamp();
        let marker = rtp.is_marker();
        let has_extension = rtp.is_extension();
        let payload = rtp.payload().ok()?;
        let payload_len = payload.len() as u32;
        let osn = (payload.len() >= 2).then(|| u16::from_be_bytes([payload[0], payload[1]]));
        drop(rtp);

        let mut info = PacketInfo {
            pt,
            ssrc,
            seqnum,
            rtp_ts,
            marker,
            payload_len,
            has_extension,
            kind: Kind::Media,
            frame: None,
            frame_start: false,
            osn: None,
        };

        if self.rtx_pts.contains(&pt) {
            let requested = osn.is_some_and(|osn| {
                self.rtx_requests
                    .iter()
                    .any(|(seqnum, t)| *seqnum == osn && now - *t <= RTX_REQUEST_MAX_AGE)
            });
            info.kind = if requested {
                Kind::RtxRepair
            } else {
                Kind::RtxPadding
            };
            info.osn = osn;
            gst::trace!(
                CAT,
                obj = element,
                "RTX packet {seqnum} (ssrc {ssrc:#x}) carrying {osn:?}: {}",
                info.kind.as_str()
            );
            return Some(info);
        }

        let delta_unit = buffer.flags().contains(gst::BufferFlags::DELTA_UNIT);
        let (uid, frame_start) =
            self.track_frame(element, ssrc, rtp_ts, payload_len, delta_unit, marker, now);
        info.frame = Some(uid);
        info.frame_start = frame_start;

        Some(info)
    }

    /// Adds a media packet to its frame. Returns the frame and whether the
    /// packet starts it.
    #[allow(clippy::too_many_arguments)]
    fn track_frame(
        &mut self,
        element: &super::Pacer,
        ssrc: u32,
        rtp_ts: u32,
        payload_len: u32,
        delta_unit: bool,
        marker: bool,
        now: Instant,
    ) -> (u64, bool) {
        let stream_idx = match self.streams.iter().position(|s| s.ssrc == ssrc) {
            Some(idx) => idx,
            None => {
                gst::debug!(CAT, obj = element, "New media stream with ssrc {ssrc:#x}");
                self.streams.push(MediaStream {
                    ssrc,
                    delta_flags_seen: false,
                    frames_since_keyframe: None,
                    avg_delta_frame_bytes: None,
                    current: None,
                });
                self.streams.len() - 1
            }
        };

        let current = self.streams[stream_idx]
            .current
            .filter(|uid| self.frame(*uid).is_some_and(|f| f.rtp_ts == rtp_ts));

        let (uid, frame_start) = match current {
            Some(uid) => (uid, false),
            None => {
                if let Some(prev) = self.streams[stream_idx].current {
                    self.finish_frame(element, stream_idx, prev);
                }

                let gop_size = self.gop_size as u64;
                let uid = self.next_frame_uid;
                self.next_frame_uid += 1;

                let stream = &mut self.streams[stream_idx];
                let first = stream.current.is_none();
                if delta_unit && !stream.delta_flags_seen {
                    gst::debug!(
                        CAT,
                        obj = element,
                        "Payloader flags delta units on ssrc {ssrc:#x}, frames without the flag \
                         are keyframes"
                    );
                    stream.delta_flags_seen = true;
                }
                // The first frame of a stream is a keyframe whether or not
                // the payloader flags delta units
                let keyframe = !delta_unit && (stream.delta_flags_seen || first);
                let requested = keyframe
                    && gop_size > 0
                    && stream
                        .frames_since_keyframe
                        .is_some_and(|n| n + 1 < gop_size);
                if keyframe {
                    match stream.frames_since_keyframe {
                        Some(n) => gst::debug!(
                            CAT,
                            obj = element,
                            "Keyframe on ssrc {ssrc:#x} (RTP timestamp {rtp_ts}), {} frames \
                             after the previous one (GOP {gop_size}): {}",
                            n + 1,
                            if requested { "requested" } else { "periodic" },
                        ),
                        None => gst::debug!(
                            CAT,
                            obj = element,
                            "First keyframe on ssrc {ssrc:#x} (RTP timestamp {rtp_ts})"
                        ),
                    }
                }
                stream.frames_since_keyframe = if keyframe {
                    Some(0)
                } else {
                    stream.frames_since_keyframe.map(|n| n + 1)
                };
                stream.current = Some(uid);

                self.frames.push_back(Frame {
                    uid,
                    rtp_ts,
                    bytes: 0,
                    packets: 0,
                    keyframe,
                    requested,
                    large: false,
                    complete: false,
                    created: now,
                });

                (uid, true)
            }
        };

        if delta_unit {
            self.streams[stream_idx].delta_flags_seen = true;
        }
        let avg = self.streams[stream_idx].avg_delta_frame_bytes;
        let frame = self.frame_mut(uid).unwrap();
        frame.bytes += u64::from(payload_len);
        frame.packets += 1;
        if !frame.keyframe
            && !frame.large
            && avg.is_some_and(|avg| frame.bytes as f64 > LARGE_FRAME_RATIO * avg)
        {
            frame.large = true;
            gst::debug!(
                CAT,
                obj = element,
                "Frame on ssrc {ssrc:#x} (RTP timestamp {rtp_ts}) is over {} bytes, more than \
                 {LARGE_FRAME_RATIO}x the average delta frame ({:.0} bytes): pacing it like a \
                 keyframe",
                frame.bytes,
                avg.unwrap_or(0.),
            );
        }

        if marker {
            self.finish_frame(element, stream_idx, uid);
        }

        (uid, frame_start)
    }

    /// Marks a frame complete once its last packet is in, and folds delta
    /// frames into the stream's average frame size
    fn finish_frame(&mut self, element: &super::Pacer, stream_idx: usize, uid: u64) {
        let avg_delta_frame_bytes = self.streams[stream_idx].avg_delta_frame_bytes;
        let Some(frame) = self.frame_mut(uid) else {
            return;
        };
        if frame.complete {
            return;
        }
        frame.complete = true;

        if frame.is_burst() {
            gst::debug!(
                CAT,
                obj = element,
                "{} with RTP timestamp {} complete: {} bytes in {} packets (average delta frame \
                 {:?} bytes)",
                if frame.keyframe {
                    "Keyframe"
                } else {
                    "Large frame"
                },
                frame.rtp_ts,
                frame.bytes,
                frame.packets,
                avg_delta_frame_bytes.map(f64::round),
            );
        }

        // Keyframes and large frames would make the average useless for
        // telling them apart
        if !frame.is_burst() {
            let bytes = frame.bytes as f64;
            let stream = &mut self.streams[stream_idx];
            stream.avg_delta_frame_bytes = Some(match stream.avg_delta_frame_bytes {
                Some(avg) => avg + DELTA_FRAME_SIZE_ALPHA * (bytes - avg),
                None => bytes,
            });
        }
    }

    /// Queues a packet: RTX ahead of media, so repairs aren't stuck behind
    /// a keyframe
    fn enqueue(&mut self, element: &super::Pacer, buffer: gst::Buffer, now: Instant) {
        let info = self.classify(element, &buffer, now);
        if info.is_none() {
            gst::log!(
                CAT,
                obj = element,
                "Not an RTP packet, pacing it without knowing what it is"
            );
        }
        self.queued_bytes += buffer.size() as u64;

        // Packets queued behind a keyframe or large frame get its longer
        // queue time, and keep it once the frame has left: shortening it
        // then would leave them overdue and send them as a burst
        let burst_frame = info
            .and_then(|info| info.frame)
            .filter(|uid| self.frame(*uid).is_some_and(Frame::is_burst));
        let queue_time = if burst_frame.is_some() || self.burst_queued() {
            self.keyframe_queue_time()
        } else {
            self.frame_queue_time()
        };
        if let Some(uid) = burst_frame {
            // A large frame is only found out a few packets in
            let keyframe_queue_time = self.keyframe_queue_time();
            for queued in self
                .media_queue
                .iter_mut()
                .rev()
                .take_while(|q| q.info.and_then(|info| info.frame) == Some(uid))
            {
                queued.deadline = queued.deadline.max(queued.enqueued + keyframe_queue_time);
            }
        }

        let queued = Queued {
            buffer,
            info,
            enqueued: now,
            deadline: now + queue_time,
        };
        if info.is_some_and(|info| info.kind != Kind::Media) {
            self.rtx_queue.push_back(queued);
        } else {
            self.media_queue.push_back(queued);
            self.media_enqueued += 1;
        }
    }

    /// Queues a serialized event behind the media packets queued so far
    fn enqueue_event(&mut self, event: gst::Event) {
        self.events.push_back((self.media_enqueued, event));
    }

    /// Drops what's queued on flush, except the sticky events that
    /// downstream needs before the next buffers
    fn flush(&mut self) {
        self.rtx_queue.clear();
        self.media_queue.clear();
        self.queued_bytes = 0;
        self.events
            .retain(|(_, event)| event.is_sticky() && event.type_() != gst::EventType::Eos);
        for (pos, _) in self.events.iter_mut() {
            *pos = 0;
        }
        self.media_enqueued = 0;
        self.media_dequeued = 0;
    }

    /// Moves the events due before the next media packet to `list`
    fn release_events(&mut self, list: &mut Vec<Output>) {
        while self
            .events
            .front()
            .is_some_and(|(pos, _)| *pos <= self.media_dequeued)
        {
            list.push(Output::Event(self.events.pop_front().unwrap().1));
        }
    }

    /// Whether a keyframe or large frame is waiting in the queue
    fn burst_queued(&self) -> bool {
        let mut last = None;
        self.media_queue
            .iter()
            .filter_map(|q| q.info.and_then(|info| info.frame))
            .any(|uid| {
                if last == Some(uid) {
                    return false;
                }
                last = Some(uid);
                self.frame(uid).is_some_and(Frame::is_burst)
            })
    }

    /// The packets the budget allows sending now, and the events due
    fn dequeue(&mut self, element: &super::Pacer, now: Instant) -> Vec<Output> {
        let elapsed = self.last_push.map_or(Duration::ZERO, |t| now - t);
        self.last_push = Some(now);

        // Speed up just enough for every packet to leave by its deadline,
        // rather than send what doesn't fit as a burst
        let mut rate = self.pacing_rate();
        let mut ahead_bits = 0.;
        for queued in self.rtx_queue.iter().chain(self.media_queue.iter()) {
            ahead_bits += queued.buffer.size() as f64 * 8.;
            let left = queued.deadline.saturating_duration_since(now).max(TICK);
            rate = rate.max(ahead_bits / left.as_secs_f64());
        }

        let speeding_up = rate > self.pacing_rate();
        if speeding_up != self.speeding_up {
            self.speeding_up = speeding_up;
            if speeding_up {
                gst::debug!(
                    CAT,
                    obj = element,
                    "Speeding up to {rate:.0} bit/s (pacing rate {:.0}) to send {} queued bytes \
                     by their deadlines (keyframe queued: {})",
                    self.pacing_rate(),
                    self.queued_bytes,
                    self.burst_queued(),
                );
            } else {
                gst::debug!(
                    CAT,
                    obj = element,
                    "Back to the pacing rate ({:.0} bit/s), {} bytes queued",
                    self.pacing_rate(),
                    self.queued_bytes,
                );
            }
        }

        let mut budget = self.budget_offset + rate * elapsed.as_secs_f64();
        let mut output = Vec::new();
        loop {
            self.release_events(&mut output);
            if budget <= 0. {
                break;
            }

            let queued = match self.rtx_queue.pop_front() {
                Some(queued) => queued,
                None => match self.media_queue.pop_front() {
                    Some(queued) => {
                        self.media_dequeued += 1;
                        queued
                    }
                    None => break,
                },
            };

            let size = queued.buffer.size() as u64;
            budget -= size as f64 * 8.;
            self.queued_bytes -= size;

            // How long a keyframe sat in the pacer, from its first packet
            // coming in to its last going out
            if let Some(frame) = queued
                .info
                .filter(|info| info.marker)
                .and_then(|info| info.frame)
                .and_then(|uid| self.frame(uid))
                .filter(|frame| frame.is_burst())
            {
                gst::debug!(
                    CAT,
                    obj = element,
                    "{} with RTP timestamp {} ({} bytes) went out over {:?}",
                    if frame.keyframe {
                        "Keyframe"
                    } else {
                        "Large frame"
                    },
                    frame.rtp_ts,
                    frame.bytes,
                    now - frame.created,
                );
            }

            if let Some(info) = queued.info.filter(|info| info.has_extension) {
                self.sent.push_back(Sent {
                    index: self.next_index,
                    info,
                    at: now,
                });
                self.next_index += 1;
            }

            output.push(Output::Buffer(queued.buffer));
        }

        // Budget left over while the queue is empty doesn't carry beyond a
        // tick's worth, or the next frame would go out as a burst
        self.budget_offset = budget.min(rate * TICK.as_secs_f64());

        if !output.is_empty() {
            gst::trace!(
                CAT,
                obj = element,
                "Sending {} items at {rate:.0} bit/s, {} bytes still queued",
                output.len(),
                self.queued_bytes,
            );
        }

        output
    }

    /// Forgets sent packets, frames and retransmission requests too old to
    /// matter for feedback or classification
    fn prune(&mut self, now: Instant) {
        while self
            .sent
            .front()
            .is_some_and(|s| now - s.at > SENT_LOG_MAX_AGE)
        {
            self.sent.pop_front();
        }

        let oldest_current = self.streams.iter().filter_map(|s| s.current).min();
        while self.frames.front().is_some_and(|f| {
            now - f.created > SENT_LOG_MAX_AGE && oldest_current.map_or(true, |uid| f.uid < uid)
        }) {
            self.frames.pop_front();
        }

        while self
            .rtx_requests
            .front()
            .is_some_and(|(_, t)| now - *t > RTX_REQUEST_MAX_AGE)
        {
            self.rtx_requests.pop_front();
        }
    }

    /// Extends a 16-bit TWCC seqnum to 64 bits, relative to the highest
    /// seen so far
    fn extend_seqnum(&mut self, seqnum: u16) -> i64 {
        let extended = match self.twcc_extended_seqnum {
            None => i64::from(seqnum),
            Some(last) => last + i64::from(seqnum.wrapping_sub(last as u16) as i16),
        };
        self.twcc_extended_seqnum = Some(
            self.twcc_extended_seqnum
                .map_or(extended, |last| last.max(extended)),
        );
        extended
    }

    /// Sent packet with the given index in the sent log
    fn sent_at(&self, index: i64) -> Option<&Sent> {
        let front = self.sent.front()?.index as i64;
        let pos = usize::try_from(index - front).ok()?;
        self.sent.get(pos)
    }

    /// (matching, mismatching) feedback packets with `offset`, among those
    /// still in the sent log
    fn check_offset(&self, packets: &[Feedback], offset: i64) -> (usize, usize) {
        packets.iter().fold((0, 0), |(matched, mismatched), fb| {
            match self.sent_at(fb.seqnum - offset) {
                Some(sent) if sent.info.pt == fb.pt && sent.info.payload_len == fb.size => {
                    (matched + 1, mismatched)
                }
                Some(_) => (matched, mismatched + 1),
                None => (matched, mismatched),
            }
        })
    }

    /// Finds where `packets` line up with the sent log, preferring the
    /// offset closest to the previous one (or 0: rtpsession numbers packets
    /// from 0, and the pacer sees all of them from the start)
    fn find_offset(&self, packets: &[Feedback]) -> Option<i64> {
        let expected = self.offset.unwrap_or(0);
        let anchor = packets.first()?;
        let fully_matches = |offset: i64| {
            let (matched, mismatched) = self.check_offset(packets, offset);
            matched > 0 && mismatched == 0
        };

        if fully_matches(expected) {
            return Some(expected);
        }

        // Full size packets all look the same, so several offsets can match
        // within a frame
        self.sent
            .iter()
            .filter(|s| s.info.pt == anchor.pt && s.info.payload_len == anchor.size)
            .map(|s| anchor.seqnum - s.index as i64)
            .filter(|offset| fully_matches(*offset))
            .min_by_key(|offset| (offset - expected).abs())
    }

    /// Annotates the packets of an `RTPTWCCPackets` event with what the
    /// pacer knows about them
    fn on_feedback(
        &mut self,
        element: &super::Pacer,
        structure: &gst::StructureRef,
    ) -> Option<gst::Structure> {
        let now = Instant::now();
        self.prune(now);

        let varray = structure.get::<glib::ValueArray>("packets").ok()?;
        let packets = varray
            .iter()
            .filter_map(|v| v.get::<gst::Structure>().ok())
            .filter_map(|s| Feedback::from_structure(&s))
            .map(|(seqnum, fb)| Feedback {
                seqnum: self.extend_seqnum(seqnum),
                ..fb
            })
            .collect::<Vec<_>>();
        if packets.is_empty() {
            return None;
        }

        let lined_up = self.offset.is_some_and(|offset| {
            let (matched, mismatched) = self.check_offset(&packets, offset);
            mismatched == 0 || matched >= 3 * mismatched
        });
        if !lined_up {
            let offset = self.find_offset(&packets);
            if offset.is_none() && self.offset.is_some() {
                gst::warning!(
                    CAT,
                    obj = element,
                    "TWCC feedback no longer lines up with the sent packets (was at offset {:?}), \
                     reporting them as unknown",
                    self.offset
                );
            } else if offset != self.offset {
                gst::debug!(
                    CAT,
                    obj = element,
                    "TWCC feedback lines up with sent packets at offset {offset:?} (was {:?})",
                    self.offset
                );
            }
            self.offset = offset;
        }

        let annotated = packets
            .iter()
            .map(|fb| {
                let info = self
                    .offset
                    .and_then(|offset| self.sent_at(fb.seqnum - offset))
                    .map(|sent| sent.info)
                    .filter(|info| info.pt == fb.pt && info.payload_len == fb.size);

                let mut s = gst::Structure::builder("RTPPacerPacket")
                    .field("twcc-seqnum", fb.seqnum)
                    .field("local-ts", fb.local_ts)
                    .field("size", fb.size)
                    .field("lost", fb.lost)
                    .field("payload-type", u32::from(fb.pt));
                if let Some(remote_ts) = fb.remote_ts {
                    s = s.field("remote-ts", remote_ts);
                }

                s = match info {
                    None => s.field("kind", "unknown"),
                    Some(info) => {
                        let mut s = s
                            .field("kind", info.kind.as_str())
                            .field("ssrc", info.ssrc)
                            .field("rtp-seqnum", u32::from(info.seqnum))
                            .field("rtp-timestamp", info.rtp_ts)
                            .field("marker", info.marker);
                        if let Some(osn) = info.osn {
                            s = s.field("osn", u32::from(osn));
                        }
                        match info.frame.and_then(|uid| self.frame(uid)) {
                            Some(frame) => s
                                .field("frame-start", info.frame_start)
                                .field("keyframe", frame.keyframe)
                                .field("keyframe-requested", frame.requested)
                                .field("large-frame", frame.large)
                                .field("frame-size", frame.bytes)
                                .field("frame-packets", frame.packets),
                            None => s,
                        }
                    }
                };

                s.build().to_send_value()
            })
            .collect::<gst::Array>();

        let (matched, _) = self
            .offset
            .map_or((0, 0), |offset| self.check_offset(&packets, offset));
        gst::log!(
            CAT,
            obj = element,
            "Feedback for {} packets (TWCC {}..={}): {matched} identified, {} lost",
            packets.len(),
            packets.first().map_or(0, |fb| fb.seqnum),
            packets.last().map_or(0, |fb| fb.seqnum),
            packets.iter().filter(|fb| fb.lost).count(),
        );

        Some(
            gst::Structure::builder("RTPPacerFeedback")
                .field("packets", annotated)
                .field("framerate", self.framerate)
                .field("gop-size", self.gop_size)
                .build(),
        )
    }
}

pub struct Pacer {
    /// Everything the chain function, srcpad task and events share
    state: Mutex<State>,

    /// Pushes paced packets, receives TWCC and retransmission events
    srcpad: gst::Pad,
    /// Receives RTP packets from rtprtxsend
    sinkpad: gst::Pad,
}

impl Pacer {
    /// Pushes dequeued buffers and events downstream, in order, stopping at
    /// the first flow error
    fn push_list(&self, list: Vec<Output>) -> Result<gst::FlowSuccess, gst::FlowError> {
        let mut res = Ok(gst::FlowSuccess::Ok);
        for output in list {
            match output {
                Output::Buffer(buf) => res = self.srcpad.push(buf),
                Output::Event(event) => {
                    let _ = self.srcpad.push_event(event);
                }
            }
            if res.is_err() {
                break;
            }
        }

        self.state.lock().unwrap().flow_return = res;

        res
    }

    /// Starts the srcpad task that dequeues every TICK on the system clock
    fn start_task(&self) -> Result<(), glib::BoolError> {
        let weak_pacer = self.obj().downgrade();
        let weak_pad = self.srcpad.downgrade();
        let clock = gst::SystemClock::obtain();
        let tick = gst::ClockTime::from_nseconds(TICK.as_nanos() as u64);

        self.state.lock().unwrap().clock_entry =
            Some(clock.new_single_shot_id(clock.time().unwrap() + tick));

        self.srcpad.start_task(move || {
            let pause = || {
                if let Some(pad) = weak_pad.upgrade() {
                    let _ = pad.pause_task();
                }
            };
            let pacer = weak_pacer
                .upgrade()
                .expect("pacer destroyed while its srcpad task is still running?");
            let imp = pacer.imp();

            let Some(clock_entry) = imp.state.lock().unwrap().clock_entry.take() else {
                gst::info!(
                    CAT,
                    obj = pacer,
                    "Pausing task as our clock entry is not set anymore"
                );
                return pause();
            };

            if let (Err(err), _) = clock_entry.wait() {
                if err != gst::ClockError::Early {
                    gst::error!(
                        CAT,
                        obj = pacer,
                        "Got error {err:?} on the clock, pausing task"
                    );
                    imp.state.lock().unwrap().flow_return = Err(gst::FlowError::Flushing);
                    return pause();
                }
            }

            let list = {
                let mut state = imp.state.lock().unwrap();
                clock
                    .single_shot_id_reinit(&clock_entry, clock.time().unwrap() + tick)
                    .unwrap();
                state.clock_entry = Some(clock_entry);
                state.dequeue(&pacer, Instant::now())
            };

            if !list.is_empty() {
                if let Err(err) = imp.push_list(list) {
                    if err != gst::FlowError::Flushing {
                        gst::error!(CAT, obj = pacer, "pause task, reason: {err:?}");
                    }
                    pause()
                }
            }
        })
    }

    /// Starts the task on activation; stops it and drops the queue on
    /// deactivation
    fn src_activatemode(&self, mode: gst::PadMode, active: bool) -> Result<(), gst::LoggableError> {
        if mode != gst::PadMode::Push {
            return Err(gst::loggable_error!(CAT, "Unsupported pad mode {mode:?}"));
        }

        if active {
            self.state.lock().unwrap().flow_return = Ok(gst::FlowSuccess::Ok);
            self.start_task()?;
        } else {
            let mut state = self.state.lock().unwrap();
            state.flow_return = Err(gst::FlowError::Flushing);
            state.flush();
            drop(state);

            self.srcpad.stop_task()?;
        }

        Ok(())
    }

    /// Flushes the queue on flush-start and keeps serialized events in order
    /// with the queued packets
    fn sink_event(&self, pad: &gst::Pad, event: gst::Event) -> bool {
        match event.type_() {
            gst::EventType::FlushStart => {
                gst::debug!(CAT, imp = self, "Flushing, dropping the queued packets");
                self.state.lock().unwrap().flush()
            }
            gst::EventType::FlushStop => (),
            // Behind the queued packets, or EOS would cut off the end of
            // the stream
            _ if event.is_serialized() => {
                let mut state = self.state.lock().unwrap();
                if event.type_() == gst::EventType::Eos {
                    gst::debug!(
                        CAT,
                        imp = self,
                        "EOS queued behind {} bytes",
                        state.queued_bytes
                    );
                }
                state.enqueue_event(event);
                return true;
            }
            _ => (),
        }

        gst::Pad::event_default(pad, Some(&*self.obj()), event)
    }

    /// Turns TWCC feedback into `packet-feedback` signals and records
    /// retransmission requests, then forwards upstream
    fn src_event(&self, pad: &gst::Pad, event: gst::Event) -> bool {
        if let Some(structure) = event.structure() {
            match structure.name().as_str() {
                "RTPTWCCPackets" => {
                    let feedback = self
                        .state
                        .lock()
                        .unwrap()
                        .on_feedback(&self.obj(), structure);
                    // Not holding the lock, handlers set properties on us
                    if let Some(feedback) = feedback {
                        self.obj()
                            .emit_by_name::<()>("packet-feedback", &[&feedback]);
                    }
                }
                "GstRTPRetransmissionRequest" => {
                    if let Ok(seqnum) = structure.get::<u32>("seqnum") {
                        gst::trace!(
                            CAT,
                            imp = self,
                            "Retransmission requested for {seqnum} (ssrc {:?})",
                            structure.get::<u32>("ssrc").ok()
                        );
                        let now = Instant::now();
                        let mut state = self.state.lock().unwrap();
                        state.prune(now);
                        state.rtx_requests.push_back((seqnum as u16, now));
                    }
                }
                _ => (),
            }
        }

        gst::Pad::event_default(pad, Some(&*self.obj()), event)
    }
}

#[glib::object_subclass]
impl ObjectSubclass for Pacer {
    const NAME: &'static str = "GstRtpPacer";
    type Type = super::Pacer;
    type ParentType = gst::Element;

    fn with_class(klass: &Self::Class) -> Self {
        let templ = klass.pad_template("sink").unwrap();
        let sinkpad = gst::Pad::builder_from_template(&templ)
            .chain_function(|_pad, parent, buffer| {
                Pacer::catch_panic_pad_function(
                    parent,
                    || Err(gst::FlowError::Error),
                    |this| {
                        let mut state = this.state.lock().unwrap();
                        state.enqueue(&this.obj(), buffer, Instant::now());
                        state.flow_return
                    },
                )
            })
            .event_function(|pad, parent, event| {
                Pacer::catch_panic_pad_function(
                    parent,
                    || false,
                    |this| this.sink_event(pad, event),
                )
            })
            .flags(gst::PadFlags::PROXY_CAPS | gst::PadFlags::PROXY_ALLOCATION)
            .build();

        let templ = klass.pad_template("src").unwrap();
        let srcpad = gst::Pad::builder_from_template(&templ)
            .event_function(|pad, parent, event| {
                Pacer::catch_panic_pad_function(parent, || false, |this| this.src_event(pad, event))
            })
            .activatemode_function(|_pad, parent, mode, active| {
                Pacer::catch_panic_pad_function(
                    parent,
                    || {
                        Err(gst::loggable_error!(
                            CAT,
                            "Panic activating src pad with mode"
                        ))
                    },
                    |this| this.src_activatemode(mode, active),
                )
            })
            .flags(gst::PadFlags::PROXY_CAPS | gst::PadFlags::PROXY_ALLOCATION)
            .build();

        Self {
            state: Default::default(),
            srcpad,
            sinkpad,
        }
    }
}

impl ObjectImpl for Pacer {
    fn constructed(&self) {
        self.parent_constructed();

        let obj = self.obj();
        obj.add_pad(&self.sinkpad).unwrap();
        obj.add_pad(&self.srcpad).unwrap();
    }

    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: Lazy<Vec<glib::ParamSpec>> = Lazy::new(|| {
            vec![
                glib::ParamSpecUInt::builder("bitrate")
                    .nick("Bitrate")
                    .blurb("Target bitrate of the stream, in bit/sec, the pacing rate is derived from")
                    .minimum(1)
                    .default_value(DEFAULT_BITRATE)
                    .mutable_playing()
                    .build(),
                glib::ParamSpecDouble::builder("pacing-factor")
                    .nick("Pacing Factor")
                    .blurb("Packets are sent at this many times the bitrate")
                    .minimum(1.)
                    .maximum(10.)
                    .default_value(DEFAULT_PACING_FACTOR)
                    .mutable_playing()
                    .build(),
                glib::ParamSpecDouble::builder("framerate")
                    .nick("Framerate")
                    .blurb("Framerate of the stream, the queue holds at most one frame interval of data (a few while a keyframe is queued)")
                    .minimum(1.)
                    .maximum(1000.)
                    .default_value(DEFAULT_FRAMERATE)
                    .mutable_playing()
                    .build(),
                glib::ParamSpecUInt::builder("gop-size")
                    .nick("GOP Size")
                    .blurb("Frames between periodic keyframes, 0 if unknown: a queued keyframe may take at most half a GOP to drain, and keyframes before the GOP is over are reported as requested")
                    .default_value(DEFAULT_GOP_SIZE)
                    .mutable_playing()
                    .build(),
                glib::ParamSpecBoxed::builder::<gst::Structure>("rtx-payload-type-map")
                    .nick("RTX Payload Type Map")
                    .blurb("Map of original payload types to RTX payload types, as rtprtxsend's payload-type-map")
                    .mutable_playing()
                    .build(),
                glib::ParamSpecUInt64::builder("queued-bytes")
                    .nick("Queued Bytes")
                    .blurb("Bytes waiting in the pacer's queue")
                    .read_only()
                    .build(),
            ]
        });

        PROPERTIES.as_ref()
    }

    fn signals() -> &'static [glib::subclass::Signal] {
        static SIGNALS: Lazy<Vec<glib::subclass::Signal>> = Lazy::new(|| {
            vec![
                /*
                 * rtppacer::packet-feedback:
                 * @feedback: an `RTPPacerFeedback` structure, its `packets`
                 * array holding one `RTPPacerPacket` per packet of the TWCC
                 * feedback, with what the pacer knows about it
                 *
                 * Emitted from the streaming thread of the feedback, without
                 * the pacer's lock held.
                 */
                glib::subclass::Signal::builder("packet-feedback")
                    .param_types([gst::Structure::static_type()])
                    .build(),
            ]
        });

        SIGNALS.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        let mut state = self.state.lock().unwrap();
        match pspec.name() {
            "bitrate" => {
                let bitrate = value.get().expect("type checked upstream");
                gst::log!(
                    CAT,
                    imp = self,
                    "Bitrate {} -> {bitrate}, pacing at {:.0} bit/s",
                    state.bitrate,
                    f64::from(bitrate) * state.pacing_factor
                );
                state.bitrate = bitrate;
            }
            "pacing-factor" => {
                state.pacing_factor = value.get().expect("type checked upstream");
                gst::info!(CAT, imp = self, "Pacing factor {}", state.pacing_factor);
            }
            "framerate" => {
                state.framerate = value.get().expect("type checked upstream");
                gst::info!(
                    CAT,
                    imp = self,
                    "Framerate {}: queue time {:?}, {:?} with a keyframe queued",
                    state.framerate,
                    state.frame_queue_time(),
                    state.keyframe_queue_time()
                );
            }
            "gop-size" => {
                state.gop_size = value.get().expect("type checked upstream");
                gst::info!(
                    CAT,
                    imp = self,
                    "GOP size {}: queue time {:?} with a keyframe queued",
                    state.gop_size,
                    state.keyframe_queue_time()
                );
            }
            "rtx-payload-type-map" => {
                state.set_rtx_pt_map(value.get().expect("type checked upstream"));
                gst::info!(
                    CAT,
                    imp = self,
                    "RTX payload types {:?} from {:?}",
                    state.rtx_pts,
                    state.rtx_pt_map
                );
            }
            _ => unimplemented!(),
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        let state = self.state.lock().unwrap();
        match pspec.name() {
            "bitrate" => state.bitrate.to_value(),
            "pacing-factor" => state.pacing_factor.to_value(),
            "framerate" => state.framerate.to_value(),
            "gop-size" => state.gop_size.to_value(),
            "rtx-payload-type-map" => state.rtx_pt_map.to_value(),
            "queued-bytes" => state.queued_bytes.to_value(),
            _ => unimplemented!(),
        }
    }
}

impl GstObjectImpl for Pacer {}

impl ElementImpl for Pacer {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static ELEMENT_METADATA: Lazy<gst::subclass::ElementMetadata> = Lazy::new(|| {
            gst::subclass::ElementMetadata::new(
                "RTP pacer",
                "Network/WebRTC/RTP/Filter",
                "Paces RTP packets, keyframe and retransmission aware, and annotates \
                 the transport-wide congestion control feedback per packet",
                "viotov <viotov@moveparallel.com>",
            )
        });

        Some(&*ELEMENT_METADATA)
    }

    fn pad_templates() -> &'static [gst::PadTemplate] {
        static PAD_TEMPLATES: Lazy<Vec<gst::PadTemplate>> = Lazy::new(|| {
            let caps = gst::Caps::builder_full()
                .structure(gst::Structure::builder("application/x-rtp").build())
                .build();

            vec![
                gst::PadTemplate::new(
                    "sink",
                    gst::PadDirection::Sink,
                    gst::PadPresence::Always,
                    &caps,
                )
                .unwrap(),
                gst::PadTemplate::new(
                    "src",
                    gst::PadDirection::Src,
                    gst::PadPresence::Always,
                    &caps,
                )
                .unwrap(),
            ]
        });

        PAD_TEMPLATES.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gst_rtp::prelude::*;

    fn init() {
        use std::sync::Once;
        static INIT: Once = Once::new();
        INIT.call_once(|| gst::init().unwrap());
    }

    /// A pacer element, for log context
    fn element() -> super::super::Pacer {
        init();
        glib::Object::new()
    }

    /// RTP packet with a header extension (so it gets a TWCC seqnum), and
    /// for RTX the sequence number it carries
    fn rtp_buffer(
        pt: u8,
        seqnum: u16,
        ts: u32,
        payload_len: u32,
        marker: bool,
        delta: bool,
        osn: Option<u16>,
    ) -> gst::Buffer {
        let mut buffer = gst::Buffer::new_rtp_with_sizes(payload_len, 0, 0).unwrap();
        {
            let buffer = buffer.get_mut().unwrap();
            let mut rtp = gst_rtp::RTPBuffer::from_buffer_writable(buffer).unwrap();
            rtp.set_payload_type(pt);
            rtp.set_seq(seqnum);
            rtp.set_timestamp(ts);
            rtp.set_ssrc(0x1234);
            rtp.set_marker(marker);
            if let Some(osn) = osn {
                rtp.payload_mut().unwrap()[..2].copy_from_slice(&osn.to_be_bytes());
            }
            rtp.add_extension_onebyte_header(1, &[0, 0]).unwrap();
            drop(rtp);
            if delta {
                buffer.set_flags(gst::BufferFlags::DELTA_UNIT);
            }
        }
        buffer
    }

    /// Queues a frame of `packets` packets of `size` bytes
    fn frame(state: &mut State, seqnum: &mut u16, ts: u32, packets: u32, size: u32, delta: bool) {
        frame_at(state, seqnum, ts, packets, size, delta, Instant::now());
    }

    /// Same, queued at `at`
    fn frame_at(
        state: &mut State,
        seqnum: &mut u16,
        ts: u32,
        packets: u32,
        size: u32,
        delta: bool,
        at: Instant,
    ) {
        for i in 0..packets {
            state.enqueue(
                &element(),
                rtp_buffer(96, *seqnum, ts, size, i + 1 == packets, delta, None),
                at,
            );
            *seqnum += 1;
        }
    }

    #[test]
    fn queue_times_follow_framerate_and_gop() {
        let mut state = State::default();
        // 10 fps, GOP 25
        assert_eq!(state.frame_queue_time(), Duration::from_millis(100));
        assert_eq!(state.keyframe_queue_time(), Duration::from_millis(300));

        // a short GOP caps the keyframe queue at half of it
        state.gop_size = 4;
        assert_eq!(state.keyframe_queue_time(), Duration::from_millis(200));

        // high framerates keep MIN_PACING_QUEUE
        state.framerate = 60.;
        state.gop_size = 0;
        assert_eq!(state.frame_queue_time(), MIN_PACING_QUEUE);
        assert_eq!(state.keyframe_queue_time(), Duration::from_millis(50));
    }

    #[test]
    fn keyframes_and_large_frames() {
        init();
        let mut state = State::default();
        let mut seqnum = 0;

        frame(&mut state, &mut seqnum, 0, 10, 1000, false);
        for i in 1..5 {
            frame(&mut state, &mut seqnum, i * 9000, 2, 1000, true);
        }
        // an unflagged intra frame, sized like one
        frame(&mut state, &mut seqnum, 5 * 9000, 8, 1000, true);
        // a keyframe before the GOP is over
        frame(&mut state, &mut seqnum, 6 * 9000, 10, 1000, false);

        let frames = state.frames.iter().collect::<Vec<_>>();
        assert!(frames[0].keyframe && !frames[0].requested);
        assert!(frames[1..5].iter().all(|f| !f.is_burst()));
        assert!(!frames[5].keyframe && frames[5].large);
        assert!(frames[6].keyframe && frames[6].requested);
        assert_eq!(state.streams[0].avg_delta_frame_bytes, Some(2000.));
        assert!(state.burst_queued());
    }

    #[test]
    fn rtx_skips_ahead_and_is_classified() {
        init();
        let mut state = State::default();
        state.set_rtx_pt_map(Some(
            gst::Structure::builder("application/x-rtp-pt-map")
                .field("96", 97u32)
                .build(),
        ));
        let mut seqnum = 0;
        frame(&mut state, &mut seqnum, 0, 3, 1000, false);

        // a NACK for packet 1: its retransmission repairs, one of packet 7
        // (never asked for) is padding
        state.rtx_requests.push_back((1, Instant::now()));
        state.enqueue(
            &element(),
            rtp_buffer(97, 0, 0, 1002, false, false, Some(1)),
            Instant::now(),
        );
        state.enqueue(
            &element(),
            rtp_buffer(97, 1, 0, 1002, false, false, Some(7)),
            Instant::now(),
        );

        let start = Instant::now();
        state.last_push = Some(start);
        let sent = state.dequeue(&element(), start + Duration::from_secs(1));
        assert_eq!(sent.len(), 5);
        let kinds = state.sent.iter().map(|s| s.info.kind).collect::<Vec<_>>();
        assert_eq!(
            kinds,
            [
                Kind::RtxRepair,
                Kind::RtxPadding,
                Kind::Media,
                Kind::Media,
                Kind::Media
            ]
        );
    }

    #[test]
    fn pacing_speeds_up_to_drain_a_keyframe_in_time() {
        init();
        let mut state = State {
            bitrate: 1_000_000,
            pacing_factor: 1.,
            ..Default::default()
        };
        let mut seqnum = 0;
        // 120 kB keyframe: 0.96 s at 1 Mbps, must be through in 300 ms
        frame(&mut state, &mut seqnum, 0, 100, 1200, false);

        let start = Instant::now();
        state.last_push = Some(start);
        let mut t = start;
        let mut sent = 0;
        while sent < 100 {
            t += TICK;
            sent += state.dequeue(&element(), t).len();
            assert!(
                t - start < Duration::from_millis(320),
                "still {} queued",
                100 - sent
            );
        }
        // and didn't go out at once
        assert!(t - start > Duration::from_millis(250));
    }

    #[test]
    fn delta_frames_behind_a_keyframe_are_not_burst_out_after_it() {
        init();
        let mut state = State {
            bitrate: 1_000_000,
            pacing_factor: 1.,
            ..Default::default()
        };
        let mut seqnum = 0;
        let start = Instant::now();
        state.last_push = Some(start);
        // 60 kB keyframe: 480 ms at 1 Mbps, must be through in 300 ms
        frame_at(&mut state, &mut seqnum, 0, 50, 1200, false, start);

        let mut t = start;
        let mut sent = 0;
        while sent < 50 + 2 * 5 {
            t += TICK;
            // Delta frames coming in while the keyframe is still queued
            for (i, at) in [100, 200].into_iter().enumerate() {
                if t - start == Duration::from_millis(at) {
                    frame_at(
                        &mut state,
                        &mut seqnum,
                        9000 * (i as u32 + 1),
                        5,
                        1000,
                        true,
                        t,
                    );
                }
            }
            let n = state.dequeue(&element(), t).len();
            // Keyframe pace is 1.6 Mbps, a packet per tick: no catching up
            // on overdue delta frames once it's out
            assert!(n <= 2, "{n} packets sent at once after {:?}", t - start);
            sent += n;
            assert!(t - start < Duration::from_millis(520));
        }
    }

    #[test]
    fn eos_waits_for_the_queued_packets() {
        init();
        let mut state = State::default();
        let mut seqnum = 0;
        frame(&mut state, &mut seqnum, 0, 3, 1000, false);
        state.enqueue_event(gst::event::Eos::new());

        let start = Instant::now();
        state.last_push = Some(start);
        // budget for a single packet: the EOS stays queued
        let out = state.dequeue(&element(), start + Duration::from_millis(1));
        assert!(matches!(out.as_slice(), [Output::Buffer(_)]));
        let out = state.dequeue(&element(), start + Duration::from_secs(1));
        assert!(matches!(
            out.as_slice(),
            [Output::Buffer(_), Output::Buffer(_), Output::Event(_)]
        ));
    }

    /// Logs packets of `sizes` as sent, with TWCC extensions
    fn sent(state: &mut State, sizes: &[u32]) {
        let now = Instant::now();
        for size in sizes {
            state.sent.push_back(Sent {
                index: state.next_index,
                info: PacketInfo {
                    pt: 96,
                    ssrc: 0,
                    seqnum: state.next_index as u16,
                    rtp_ts: 0,
                    marker: false,
                    payload_len: *size,
                    has_extension: true,
                    kind: Kind::Media,
                    frame: None,
                    frame_start: false,
                    osn: None,
                },
                at: now,
            });
            state.next_index += 1;
        }
    }

    /// TWCC feedback for packets of `sizes`, starting at TWCC seqnum `first`
    fn feedback(first: i64, sizes: &[u32]) -> Vec<Feedback> {
        sizes
            .iter()
            .enumerate()
            .map(|(i, size)| Feedback {
                seqnum: first + i as i64,
                local_ts: 0,
                remote_ts: Some(0),
                pt: 96,
                size: *size,
                lost: false,
            })
            .collect()
    }

    #[test]
    fn feedback_lines_up_with_sent_packets() {
        let mut state = State::default();
        sent(&mut state, &[100, 1188, 1188, 1188, 500, 200, 1188, 300]);

        // numbered from 0, as rtpsession does
        assert_eq!(state.find_offset(&feedback(2, &[1188, 1188, 500])), Some(0));

        // rtpsession numbered 3 packets the pacer never saw
        assert_eq!(
            state.find_offset(&feedback(7, &[1188, 1188, 500, 200])),
            Some(5)
        );

        // nothing like it was sent
        assert_eq!(state.find_offset(&feedback(0, &[42, 43])), None);
    }

    #[test]
    fn seqnums_extend_across_wraparound() {
        let mut state = State::default();
        assert_eq!(state.extend_seqnum(65534), 65534);
        assert_eq!(state.extend_seqnum(1), 65537);
        // late feedback for a packet before the wrap
        assert_eq!(state.extend_seqnum(65535), 65535);
        assert_eq!(state.extend_seqnum(2), 65538);
    }
}
