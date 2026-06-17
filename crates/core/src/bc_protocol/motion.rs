use super::{BcCamera, Error, Result};
use crate::bc::{model::*, xml::*};
use std::time::{Duration, Instant};
use tokio::sync::broadcast::{
    channel as broadcast_channel, Receiver as BroadcastReceiver, Sender as BroadcastSender,
};
use tokio::sync::mpsc::{channel, error::TryRecvError, Receiver};
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

/// Motion Status that the callback can send
#[derive(Clone, Copy, Debug)]
pub enum MotionStatus {
    /// Sent when motion is first detected
    Start(Instant),
    /// Sent when motion stops
    Stop(Instant),
    /// Sent when an Alarm about something other than motion was received
    NoChange(Instant),
}

/// A single decoded event from the camera's alarm/motion stream.
///
/// A single camera message can yield several of these: a `"visitor,MD"` status
/// produces both a [`Doorbell`](DecodedAlarm::Doorbell) and a
/// [`MotionStart`](DecodedAlarm::MotionStart). They are returned in order and
/// never collapsed to a single outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodedAlarm {
    /// A motion-type status began (`"MD"`, `"PIR"`, an `AItype`, ...)
    MotionStart,
    /// All statuses cleared (`"none"`) — motion stopped
    MotionStop,
    /// A doorbell / visitor press
    Doorbell,
}

/// Decode the alarm events in `list` that target `channel_id`.
///
/// The camera reports alarm status on the existing motion stream. A status of
/// `"visitor"` is a doorbell press, `"none"` is motion clearing, and any other
/// non-empty status (or a non-`"none"` `AItype`) is motion. Statuses may be
/// comma separated (e.g. `"visitor,MD"`), in which case every component is
/// decoded independently. Events on other channels are ignored.
///
/// At most one motion outcome is emitted per `AlarmEvent` (start takes
/// precedence over stop, matching the previous "any active status = motion"
/// semantics), but doorbell presses are always surfaced alongside it.
pub fn decode_alarm_events(list: &AlarmEventList, channel_id: u8) -> Vec<DecodedAlarm> {
    let mut decoded = Vec::new();
    for event in &list.alarm_events {
        if event.channel_id != channel_id {
            continue;
        }
        let mut motion_start = false;
        let mut motion_stop = false;
        for token in event.status.split(',') {
            match token.trim() {
                "" => {}
                "visitor" => decoded.push(DecodedAlarm::Doorbell),
                "none" => motion_stop = true,
                _ => motion_start = true,
            }
        }
        // A non-"none" AItype is motion, preserving the existing semantics where
        // an AI detection counts even if the raw status is "none".
        if event.ai_type.as_deref().is_some_and(|ai| ai != "none") {
            motion_start = true;
        }
        if motion_start {
            decoded.push(DecodedAlarm::MotionStart);
        } else if motion_stop {
            decoded.push(DecodedAlarm::MotionStop);
        }
    }
    decoded
}

/// A handle on current motion related events comming from the camera
///
/// When this object is dropped the motion events are stopped
pub struct MotionData {
    handle: JoinSet<Result<()>>,
    cancel: CancellationToken,
    rx: Receiver<Result<MotionStatus>>,
    last_update: MotionStatus,
    doorbell: BroadcastSender<Instant>,
}

impl MotionData {
    /// Subscribe to doorbell ("visitor") presses decoded from the same alarm
    /// stream that drives motion.
    ///
    /// Doorbell presses are discrete events rather than durable state, so they
    /// are delivered over a broadcast channel separate from the motion methods.
    /// The returned receiver only sees presses that arrive after it is created.
    pub fn doorbell(&self) -> BroadcastReceiver<Instant> {
        self.doorbell.subscribe()
    }

    /// Get if motion has been detected. Returns None if
    /// no motion data has yet been recieved from the camera
    ///
    /// An error is raised if the motion connection to the camera is dropped
    pub fn motion_detected(&mut self) -> Result<Option<bool>> {
        self.consume_motion_events()?;
        Ok(match &self.last_update {
            MotionStatus::Start(_) => Some(true),
            MotionStatus::Stop(_) => Some(false),
            MotionStatus::NoChange(_) => None,
        })
    }

    /// Get if motion has been detected within given duration. Returns None if
    /// no motion data has yet been recieved from the camera
    ///
    /// An error is raised if the motion connection to the camera is dropped
    pub fn motion_detected_within(&mut self, duration: Duration) -> Result<Option<bool>> {
        self.consume_motion_events()?;
        Ok(match &self.last_update {
            MotionStatus::Start(_) => Some(true),
            MotionStatus::Stop(time) => Some((Instant::now() - *time) < duration),
            MotionStatus::NoChange(_) => None,
        })
    }

    /// Consume the motion events diretly
    ///
    /// An error is raised if the motion connection to the camera is dropped
    pub fn consume_motion_events(&mut self) -> Result<Vec<MotionStatus>> {
        let mut results: Vec<MotionStatus> = vec![];
        loop {
            match self.rx.try_recv() {
                Ok(motion) => results.push(motion?),
                Err(TryRecvError::Empty) => break,
                Err(e) => return Err(Error::from(e)),
            }
        }
        if let Some(last) = results.last() {
            self.last_update = *last;
        }
        Ok(results)
    }

    /// Await a new motion event
    ///
    ///
    pub async fn next_motion(&mut self) -> Result<MotionStatus> {
        let motions = self.consume_motion_events()?;
        if let Some(last) = motions.last() {
            Ok(*last)
        } else if let Some(moition) = self.rx.recv().await {
            let moition = moition?;
            self.last_update = moition;
            Ok(moition)
        } else {
            Err(Error::Other("Motion dropped"))
        }
    }

    /// Wait for the motion to stop
    ///
    /// It must be stopped for at least the given duration
    pub async fn await_stop(&mut self, duration: Duration) -> Result<()> {
        let motions = self.consume_motion_events()?;
        let mut last_motion = motions.last().copied();
        loop {
            if let Some(MotionStatus::Stop(time)) = last_motion {
                // In stop state
                if duration.is_zero() || (Instant::now() - time) > duration {
                    return Ok(());
                } else {
                    // Schedule a sleep or wait for motion to start
                    let remaining_sleep = duration - (Instant::now() - time);
                    let result = tokio::select! {
                        _ = tokio::time::sleep(remaining_sleep) => {None},
                        v = async {
                            loop {
                                match self.next_motion().await {
                                    n @ Ok(MotionStatus::Start(_)) => {return n;},
                                    n @ Err(_) => {return n;},
                                    _ => {continue;}
                                }
                            }
                        } => {Some(v)}
                    };
                    if let Some(v) = result {
                        v?;
                    } else {
                        return Ok(());
                    }
                }
            }
            last_motion = Some(self.next_motion().await?);
        }
    }

    /// Wait for the motion to start
    ///
    /// The motion must have a minimum duration as given
    pub async fn await_start(&mut self, duration: Duration) -> Result<()> {
        let motions = self.consume_motion_events()?;
        let mut last_motion = motions.last().copied();
        loop {
            if let Some(MotionStatus::Start(time)) = last_motion {
                // In start state
                if duration.is_zero() || (Instant::now() - time) > duration {
                    return Ok(());
                } else {
                    // Schedule a sleep or wait for motion to stop
                    let result = tokio::select! {
                        _ = tokio::time::sleep(duration - (Instant::now() - time)) => {None},
                        v = async {
                            loop {
                                match self.next_motion().await {
                                    n @ Ok(MotionStatus::Stop(_)) => {return n;},
                                    n @ Err(_) => {return n;},
                                    _ => {continue;}
                                }
                            }
                        } => {Some(v)}
                    };
                    if let Some(v) = result {
                        v?;
                    } else {
                        return Ok(());
                    }
                }
            }
            last_motion = Some(self.next_motion().await?);
        }
    }
}

impl BcCamera {
    /// This message tells the camera to send the motion events to us
    /// Which are the recieved on msgid 33
    async fn start_motion_query(&self) -> Result<u16> {
        self.has_ability_rw("motion").await?;
        let connection = self.get_connection();

        let msg_num = self.new_message_num();
        let mut sub = connection.subscribe(MSG_ID_MOTION_REQUEST, msg_num).await?;
        let msg = Bc {
            meta: BcMeta {
                msg_id: MSG_ID_MOTION_REQUEST,
                channel_id: self.channel_id,
                msg_num,
                stream_type: 0,
                response_code: 0,
                class: 0x6414,
            },
            body: BcBody::ModernMsg(ModernMsg {
                ..Default::default()
            }),
        };

        sub.send(msg).await?;

        let msg = sub.recv().await?;

        if let BcMeta {
            response_code: 200, ..
        } = msg.meta
        {
            Ok(msg_num)
        } else {
            Err(Error::UnintelligibleReply {
                reply: std::sync::Arc::new(Box::new(msg)),
                why: "The camera did not accept the request to start motion",
            })
        }
    }

    /// This returns a data structure which can be used to
    /// query motion events
    pub async fn listen_on_motion(&self) -> Result<MotionData> {
        self.start_motion_query().await?;

        let connection = self.get_connection();

        // After start_motion_query (MSG_ID 31) the camera sends motion messages
        // when whenever motion is detected.
        let (tx, rx) = channel(20);
        let (doorbell, _) = broadcast_channel(50);
        let doorbell_task = doorbell.clone();

        let mut set = JoinSet::new();
        let channel_id = self.channel_id;
        let cancel = CancellationToken::new();
        let thread_cancel = cancel.clone();
        set.spawn(async move {
            tokio::select! {
                _ = thread_cancel.cancelled() => Result::Ok(()),
                v = async {
                    let mut sub = connection.subscribe_to_id(MSG_ID_MOTION).await?;

                    loop {
                        tokio::task::yield_now().await;
                        let msg = sub.recv().await;
                        let status = match msg {
                            Ok(motion_msg) => {
                                if let BcBody::ModernMsg(ModernMsg {
                                    payload:
                                        Some(BcPayloads::BcXml(BcXml {
                                            alarm_event_list: Some(alarm_event_list),
                                            ..
                                        })),
                                    ..
                                }) = motion_msg.body
                                {
                                    let decoded =
                                        decode_alarm_events(&alarm_event_list, channel_id);
                                    // Doorbell presses are discrete events: emit
                                    // one per press on the broadcast channel. A
                                    // send error just means nobody is listening.
                                    for _ in decoded
                                        .iter()
                                        .filter(|d| matches!(d, DecodedAlarm::Doorbell))
                                    {
                                        let _ = doorbell_task.send(Instant::now());
                                    }
                                    // Motion keeps its single-state-per-message
                                    // semantics for RTSP gating: any motion start
                                    // wins, else a stop, else no change.
                                    let result = if decoded
                                        .iter()
                                        .any(|d| matches!(d, DecodedAlarm::MotionStart))
                                    {
                                        MotionStatus::Start(Instant::now())
                                    } else if decoded
                                        .iter()
                                        .any(|d| matches!(d, DecodedAlarm::MotionStop))
                                    {
                                        MotionStatus::Stop(Instant::now())
                                    } else {
                                        MotionStatus::NoChange(Instant::now())
                                    };
                                    Ok(result)
                                } else {
                                    Ok(MotionStatus::NoChange(Instant::now()))
                                }
                            }
                            // On connection drop we stop
                            Err(e) => Err(e),
                        };

                        // A subscription error means the connection/subscription is
                        // gone. Forward it once so the consumer learns of the drop,
                        // then stop: re-polling a closed subscription returns
                        // immediately every iteration, which busy-spins this task and
                        // floods the channel with the same error.
                        let connection_lost = status.is_err();

                        if tx.send(status).await.is_err() {
                            // Motion reciever has been dropped
                            break;
                        }

                        if connection_lost {
                            break;
                        }
                    }
                    Ok(())
                } => v,
            }
        });

        Ok(MotionData {
            handle: set,
            cancel,
            rx,
            last_update: MotionStatus::NoChange(Instant::now()),
            doorbell,
        })
    }
}

impl Drop for MotionData {
    fn drop(&mut self) {
        log::trace!("Drop MotionData");
        self.cancel.cancel();
        let mut handle = std::mem::take(&mut self.handle);
        let _gt = tokio::runtime::Handle::current().enter();
        tokio::task::spawn(async move {
            while handle.join_next().await.is_some() {}
            log::trace!("Dropped MotionData");
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bc::xml::{AlarmEvent, AlarmEventList};

    fn event(channel_id: u8, status: &str) -> AlarmEvent {
        AlarmEvent {
            channel_id,
            status: status.to_string(),
            ..Default::default()
        }
    }

    fn list(events: Vec<AlarmEvent>) -> AlarmEventList {
        AlarmEventList {
            alarm_events: events,
            ..Default::default()
        }
    }

    #[test]
    fn visitor_decodes_to_doorbell_only() {
        let decoded = decode_alarm_events(&list(vec![event(0, "visitor")]), 0);
        assert_eq!(decoded, vec![DecodedAlarm::Doorbell]);
    }

    #[test]
    fn md_decodes_to_motion_start() {
        let decoded = decode_alarm_events(&list(vec![event(0, "MD")]), 0);
        assert_eq!(decoded, vec![DecodedAlarm::MotionStart]);
    }

    #[test]
    fn pir_decodes_to_motion_start() {
        let decoded = decode_alarm_events(&list(vec![event(0, "PIR")]), 0);
        assert_eq!(decoded, vec![DecodedAlarm::MotionStart]);
    }

    #[test]
    fn none_decodes_to_motion_stop() {
        let decoded = decode_alarm_events(&list(vec![event(0, "none")]), 0);
        assert_eq!(decoded, vec![DecodedAlarm::MotionStop]);
    }

    #[test]
    fn comma_separated_visitor_and_md_are_not_collapsed() {
        // A doorbell press that arrives alongside motion in a single status
        // string must surface BOTH a doorbell event and a motion start.
        let decoded = decode_alarm_events(&list(vec![event(0, "visitor,MD")]), 0);
        assert_eq!(
            decoded,
            vec![DecodedAlarm::Doorbell, DecodedAlarm::MotionStart]
        );
    }

    #[test]
    fn multiple_alarm_events_all_delivered() {
        // Several AlarmEvents in one message must each be decoded, not collapsed
        // to the last one.
        let decoded = decode_alarm_events(&list(vec![event(0, "visitor"), event(0, "MD")]), 0);
        assert_eq!(
            decoded,
            vec![DecodedAlarm::Doorbell, DecodedAlarm::MotionStart]
        );
    }

    #[test]
    fn events_on_other_channels_are_ignored() {
        let decoded = decode_alarm_events(&list(vec![event(1, "visitor"), event(2, "MD")]), 0);
        assert!(decoded.is_empty());
    }

    #[test]
    fn ai_type_counts_as_motion_start() {
        // Preserve existing motion semantics: a non-"none" AItype is motion even
        // when the status itself is "none".
        let mut ev = event(0, "none");
        ev.ai_type = Some("people".to_string());
        let decoded = decode_alarm_events(&list(vec![ev]), 0);
        assert_eq!(decoded, vec![DecodedAlarm::MotionStart]);
    }
}
