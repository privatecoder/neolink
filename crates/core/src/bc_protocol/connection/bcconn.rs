use super::BcSubscription;
use crate::{bc::model::*, Error, Result};
use futures::future::BoxFuture;
use futures::sink::{Sink, SinkExt};
use futures::stream::{Stream, StreamExt};
use log::*;
use std::collections::btree_map::Entry;
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio::sync::mpsc::{channel, error::TrySendError, Sender};
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;

use tokio::{sync::RwLock, task::JoinSet};

type MsgHandler = dyn 'static + Send + Sync + for<'a> Fn(&'a Bc) -> BoxFuture<'a, Option<Bc>>;

#[derive(Default)]
struct Subscriber {
    /// Subscribers based on their ID and their num
    /// First filtered by ID then number
    /// If num is None it will be upgraded to a Some based on the number the
    /// camera assigns
    num: BTreeMap<u32, BTreeMap<Option<u16>, Sender<Result<Bc>>>>,
    /// Subscribers based on their ID
    id: BTreeMap<u32, Arc<MsgHandler>>,
}

pub(crate) type BcConnSink = Box<dyn Sink<Bc, Error = Error> + Send + Sync + Unpin>;
pub(crate) type BcConnSource = Box<dyn Stream<Item = Result<Bc>> + Send + Sync + Unpin>;

/// A shareable connection to a camera.  Handles serialization of messages.  To send/receive, call
/// .[subscribe()] with a message number.  You can use the BcSubscription to send or receive only
/// messages with that number; each incoming message is routed to its appropriate subscriber.
///
/// There can be only one subscriber per kind of message at a time.
pub struct BcConnection {
    sink: Sender<Result<Bc>>,
    poll_commander: Sender<PollCommand>,
    rx_thread: RwLock<JoinSet<Result<()>>>,
    cancel: CancellationToken,
}

impl BcConnection {
    pub async fn new(mut sink: BcConnSink, mut source: BcConnSource) -> Result<BcConnection> {
        let (sinker, sinker_rx) = channel::<Result<Bc>>(500);
        let cancel = CancellationToken::new();

        // Raised from 200 -> 1000 for parity with the now non-blocking poll loop:
        // the poller never awaits a slow subscriber, so the only place backpressure
        // can build is this command queue; a deeper queue tolerates short bursts of
        // camera traffic without dropping inbound packets at the source stream.
        let (poll_commander, poll_commanded) = channel(1000);
        let mut poller = Poller {
            subscribers: Default::default(),
            sink: sinker.clone(),
            reciever: ReceiverStream::new(poll_commanded),
            last_full_warn: None,
            dropped_full: 0,
        };

        let mut rx_thread = JoinSet::<Result<()>>::new();
        let thread_poll_commander = poll_commander.clone();
        let thread_cancel = cancel.clone();
        rx_thread.spawn(async move {
            tokio::select! {
                _ = thread_cancel.cancelled() => {
                    Result::Ok(())
                },
                v = async {
                    let sender = thread_poll_commander;
                    while let Some(bc) = source.next().await {
                        sender.send(PollCommand::Bc(Box::new(bc))).await?;
                    }
                    Result::Ok(())
                } => v
            }
        });

        let thread_cancel = cancel.clone();
        rx_thread.spawn(async move {
            tokio::select! {
                _ = thread_cancel.cancelled() => Result::Ok(()),
                v = async {
                    let mut stream = ReceiverStream::new(sinker_rx);
                    while let Some(packet) = stream.next().await {
                        sink.send(packet?).await?;
                    }
                    Ok(())
                } => v
            }
        });

        let thread_cancel = cancel.clone();
        rx_thread.spawn(async move {
            tokio::select! {
                _ = thread_cancel.cancelled() => Result::Ok(()),
                v = async {
                    // `poller.run()` loops internally until its command channel
                    // either errors or is exhausted. An `Ok(())` return means the
                    // channel closed (all senders dropped) — i.e. the connection is
                    // gone and no further commands will ever arrive. Re-looping here
                    // would re-enter `run()`, whose `reciever.next().await` is now
                    // immediately `Ready(None)` forever: a single poll spins without
                    // yielding, pinning a core AND starving the `select!` so even
                    // `thread_cancel` can't stop it. So treat `Ok(())` as terminal.
                    let res = poller.run().await;
                    trace!("Polling has ended: {res:?}");
                    res
                }=> v
            }
        });

        Ok(BcConnection {
            sink: sinker,
            poll_commander,
            rx_thread: RwLock::new(rx_thread),
            cancel,
        })
    }

    pub(super) async fn send(&self, bc: Bc) -> crate::Result<()> {
        self.sink.send(Ok(bc)).await?;
        Ok(())
    }

    pub async fn subscribe(&self, msg_id: u32, msg_num: u16) -> Result<BcSubscription<'_>> {
        let (tx, rx) = channel(500);
        self.poll_commander
            .send(PollCommand::AddSubscriber(msg_id, Some(msg_num), tx))
            .await?;
        Ok(BcSubscription::new(rx, Some(msg_num as u32), self))
    }

    /// Some messages are initiated by the camera. This creates a handler for them
    /// It requires a closure that will be used to handle the message
    /// and return either None or Some(Bc) reply
    pub async fn handle_msg<T>(&self, msg_id: u32, handler: T) -> Result<()>
    where
        T: 'static + Send + Sync + for<'a> Fn(&'a Bc) -> BoxFuture<'a, Option<Bc>>,
    {
        self.poll_commander
            .send(PollCommand::AddHandler(msg_id, Arc::new(handler)))
            .await?;
        Ok(())
    }

    /// Some times we want to wait for a reply on a new message ID
    /// to do this we wait for the next packet with a certain ID
    /// grab it's message ID and then subscribe to that ID
    ///
    /// The command Snap that grabs a jpeg payload is an example of this
    ///
    /// This function creates a temporary handle to grab this single message
    pub async fn subscribe_to_id(&self, msg_id: u32) -> Result<BcSubscription<'_>> {
        let (tx, rx) = channel(500);
        self.poll_commander
            .send(PollCommand::AddSubscriber(msg_id, None, tx))
            .await?;
        Ok(BcSubscription::new(rx, None, self))
    }

    pub(crate) async fn join(&self) -> Result<()> {
        let mut locked_threads = self.rx_thread.write().await;
        while let Some(res) = locked_threads.join_next().await {
            match res {
                Err(e) => {
                    locked_threads.abort_all();
                    return Err(e.into());
                }
                Ok(Err(e)) => {
                    locked_threads.abort_all();
                    return Err(e);
                }
                Ok(Ok(())) => {}
            }
        }
        Ok(())
    }

    pub async fn shutdown(&self) -> Result<()> {
        let _ = self.poll_commander.send(PollCommand::Disconnect).await;
        self.cancel.cancel();
        let mut locked_threads = self.rx_thread.write().await;
        while locked_threads.join_next().await.is_some() {}
        Ok(())
    }
}

impl Drop for BcConnection {
    fn drop(&mut self) {
        log::trace!("Drop BcConnection");
        self.cancel.cancel();

        let poll_commander = self.poll_commander.clone();
        let _gt = tokio::runtime::Handle::current().enter();
        let mut threads = std::mem::take(&mut self.rx_thread);
        tokio::task::spawn(async move {
            let _ = poll_commander.send(PollCommand::Disconnect).await;
            let locked_threads = threads.get_mut();
            while locked_threads.join_next().await.is_some() {}
            log::trace!("Dropped BcConnection");
        });
    }
}

enum PollCommand {
    Bc(Box<Result<Bc>>),
    AddHandler(u32, Arc<MsgHandler>),
    AddSubscriber(u32, Option<u16>, Sender<Result<Bc>>),
    Disconnect,
}

impl std::fmt::Debug for PollCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PollCommand::Bc(_) => f.write_str("PollCommand::Bc"),
            PollCommand::AddHandler(_, _) => f.write_str("PollCommand::AddHandler"),
            PollCommand::AddSubscriber(_, _, _) => f.write_str("PollCommand::AddSubscriber"),
            PollCommand::Disconnect => f.write_str("PollCommand::Disconnect"),
        }
    }
}

struct Poller {
    subscribers: Subscriber,
    sink: Sender<Result<Bc>>,
    reciever: ReceiverStream<PollCommand>,
    /// Last time we emitted a "subscriber channel full" warning. Delivery to
    /// subscribers is non-blocking (see `run`), so a stalled consumer makes us
    /// drop its messages; this rate-limits the resulting log spam.
    last_full_warn: Option<std::time::Instant>,
    /// Count of messages dropped (across all subscribers) since the last warning.
    dropped_full: u64,
}

impl Poller {
    async fn run(&mut self) -> Result<()> {
        let cancel = CancellationToken::new();
        let _dropguard = cancel.clone().drop_guard();
        while let Some(command) = self.reciever.next().await {
            // Clean Up subscribers
            self.subscribers
                .num
                .iter_mut()
                .for_each(|(_, channels)| channels.retain(|_, channel| !channel.is_closed()));
            self.subscribers
                .num
                .retain(|_, channels| !channels.is_empty());
            // Handle the command
            match command {
                PollCommand::Bc(boxed_response) => {
                    match *boxed_response {
                        Ok(response) => {
                            let msg_id = response.meta.msg_id;
                            let msg_num = response.meta.msg_num;
                            log::trace!(
                                "Looking for ID: {} with num: {}, in {:?} and {:?}",
                                msg_id,
                                msg_num,
                                self.subscribers.id.keys().to_owned(),
                                self.subscribers
                                    .num
                                    .iter()
                                    .map(|(k, v)| (k, v.keys()))
                                    .collect::<Vec<_>>(),
                            );
                            match (
                                self.subscribers.id.get(&msg_id),
                                self.subscribers.num.get_mut(&msg_id), // Both filter first on ID
                            ) {
                                (Some(occ), _) => {
                                    log::trace!("Calling ID callback");
                                    let occ = occ.clone();
                                    let sink = self.sink.clone();
                                    // Move this on another thread coz I have NO idea
                                    // how long the callback will run for
                                    // and we must NOT hang
                                    let cancel = cancel.clone();
                                    tokio::task::spawn(async move {
                                        tokio::select! {
                                            _ = cancel.cancelled() => Result::Ok(()),
                                            v = occ(&response) => {
                                                if let Some(reply) = v {
                                                    assert!(reply.meta.msg_num == response.meta.msg_num);
                                                    sink.send(Ok(reply)).await?;
                                                }
                                                Result::Ok(())
                                            }
                                        }
                                    });
                                    log::trace!("Called ID callback");
                                }
                                (None, Some(occ)) => {
                                    let sender = if let Some(sender) =
                                        occ.get(&Some(msg_num)).filter(|a| !a.is_closed()).cloned()
                                    {
                                        // Connection with id exists and is not closed
                                        Some(sender)
                                    } else if let Some(sender) = occ.get(&None).cloned() {
                                        // Upgrade a None to a known MsgID
                                        occ.remove(&None);
                                        occ.insert(Some(msg_num), sender.clone());
                                        Some(sender)
                                    } else if occ
                                        .get(&Some(msg_num))
                                        .map(|a| a.is_closed())
                                        .unwrap_or(false)
                                    {
                                        // Connection is closed and there is no None to replace it
                                        // Remove it for cleanup and report no sender
                                        occ.remove(&Some(msg_num));
                                        None
                                    } else {
                                        None
                                    };
                                    if let Some(sender) = sender {
                                        // Non-blocking delivery: the poll loop must NEVER await a
                                        // single consumer. A slow/stalled subscriber (e.g. an
                                        // overwhelmed RTSP client's video subscription) would
                                        // otherwise block this loop and starve camera
                                        // keepalive/control traffic, causing the camera to drop
                                        // the session and forcing a reconnect cycle. So we
                                        // `try_send` and, when the subscriber's channel is full,
                                        // drop the message for that subscriber (rate-limited warn).
                                        //
                                        // Dropping is NOT keyframe-aware here: at this layer the
                                        // payload is an opaque `Bc` message, so we cannot cheaply
                                        // tell a keyframe from a P-frame. Keyframe-aware dropping
                                        // already happens downstream in the RTSP relay
                                        // (`drop_until_keyframe`); here we only guarantee the poll
                                        // loop never blocks.
                                        match sender.try_send(Ok(response)) {
                                            Ok(()) => {
                                                trace!(
                                                    "Remaining: {} of {} message space for {} (ID: {})",
                                                    sender.capacity(),
                                                    sender.max_capacity(),
                                                    &msg_num,
                                                    &msg_id
                                                );
                                            }
                                            Err(TrySendError::Full(_)) => {
                                                self.dropped_full += 1;
                                                let now = std::time::Instant::now();
                                                let should_warn = self
                                                    .last_full_warn
                                                    .map(|t| {
                                                        now.duration_since(t)
                                                            >= std::time::Duration::from_secs(5)
                                                    })
                                                    .unwrap_or(true);
                                                if should_warn {
                                                    warn!(
                                                        "Subscriber channel full for num {} (ID: {}); dropped {} message(s) to keep the poll loop responsive (camera keepalive/control must not stall)",
                                                        &msg_num, &msg_id, self.dropped_full
                                                    );
                                                    self.last_full_warn = Some(now);
                                                    self.dropped_full = 0;
                                                }
                                            }
                                            Err(TrySendError::Closed(_)) => {
                                                // Subscriber went away; it is removed from the map
                                                // at the top of the next loop iteration.
                                                trace!(
                                                    "Subscriber channel closed for num {} (ID: {})",
                                                    &msg_num,
                                                    &msg_id
                                                );
                                            }
                                        }
                                    } else {
                                        trace!(
                                            "Ignoring uninteresting message id {} (number: {})",
                                            msg_id,
                                            msg_num
                                        );
                                        trace!("Contents: {:?}", response);
                                    }
                                }
                                (None, None) => {
                                    trace!(
                                        "Ignoring uninteresting message id {} (number: {})",
                                        msg_id,
                                        msg_num
                                    );
                                    trace!("Contents: {:?}", response);
                                }
                            }
                        }
                        Err(e) => {
                            // Terminal: broadcast the error and tear down. Use try_send so a
                            // full subscriber channel can't block teardown either; any
                            // subscriber that misses this will observe the channel close.
                            for sub in self.subscribers.num.values() {
                                for sender in sub.values() {
                                    let _ = sender.try_send(Err(e.clone()));
                                }
                            }
                            self.subscribers.num.clear();
                            self.subscribers.id.clear();
                            return Err(e);
                        }
                    }
                }
                PollCommand::AddHandler(msg_id, handler) => {
                    match self.subscribers.id.entry(msg_id) {
                        Entry::Vacant(vac_entry) => {
                            vac_entry.insert(handler);
                        }
                        Entry::Occupied(_) => {
                            return Err(Error::SimultaneousSubscriptionId { msg_id });
                        }
                    };
                }
                PollCommand::AddSubscriber(msg_id, msg_num, tx) => {
                    match self
                        .subscribers
                        .num
                        .entry(msg_id)
                        .or_default()
                        .entry(msg_num)
                    {
                        Entry::Vacant(vac_entry) => {
                            vac_entry.insert(tx);
                        }
                        Entry::Occupied(mut occ_entry) => {
                            if occ_entry.get().is_closed() {
                                occ_entry.insert(tx);
                            } else {
                                // log::error!("Failed to subscribe in bcconn to {:?} for {:?}", msg_num, msg_id);
                                let _ = tx
                                    .send(Err(Error::SimultaneousSubscription { msg_num }))
                                    .await;
                            }
                        }
                    };
                }
                PollCommand::Disconnect => {
                    return Err(Error::ConnectionShutdown);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bc::model::{Bc, BcBody, BcMeta, ModernMsg};
    use tokio::time::{timeout, Duration};

    fn make_bc(msg_id: u32, msg_num: u16) -> Bc {
        Bc {
            meta: BcMeta {
                msg_id,
                channel_id: 0,
                stream_type: 0,
                response_code: 0,
                msg_num,
                class: 0x6614,
            },
            body: BcBody::ModernMsg(ModernMsg::default()),
        }
    }

    /// Regression test for the keepalive-starvation bug (upstream #399): a
    /// subscriber whose channel is full must NOT block the poll loop. We feed
    /// many messages to an undrained (capacity-1) subscriber and assert that
    /// `Poller::run` still drains its command queue and returns promptly. With
    /// the old blocking `sender.send().await`, the second message would wedge
    /// the loop forever and the timeout below would fire.
    #[tokio::test]
    async fn poller_does_not_block_on_full_subscriber() {
        let (cmd_tx, cmd_rx) = channel::<PollCommand>(1000);
        let (sink_tx, _sink_rx) = channel::<Result<Bc>>(500);

        let mut poller = Poller {
            subscribers: Default::default(),
            sink: sink_tx,
            reciever: ReceiverStream::new(cmd_rx),
            last_full_warn: None,
            dropped_full: 0,
        };

        let msg_id = 42u32;
        let msg_num = 7u16;

        // A capacity-1 subscriber channel we deliberately never drain. Hold the
        // receiver so the channel stays *open* (not closed) -> the Full path,
        // not the Closed path, is exercised.
        let (sub_tx, mut sub_rx) = channel::<Result<Bc>>(1);
        cmd_tx
            .send(PollCommand::AddSubscriber(msg_id, Some(msg_num), sub_tx))
            .await
            .unwrap();

        // Far more messages than the subscriber can hold.
        for _ in 0..100 {
            cmd_tx
                .send(PollCommand::Bc(Box::new(Ok(make_bc(msg_id, msg_num)))))
                .await
                .unwrap();
        }
        // Close the command queue so run() terminates once drained.
        drop(cmd_tx);

        // Must complete well within the timeout; a blocked poll loop never would.
        let res = timeout(Duration::from_secs(5), poller.run()).await;
        assert!(
            res.is_ok(),
            "Poller::run blocked on a full subscriber channel (keepalive-starvation regression)"
        );
        assert!(res.unwrap().is_ok(), "Poller::run returned an error");

        // Exactly one message made it into the capacity-1 channel; the rest were
        // dropped non-blockingly rather than wedging the loop.
        let mut received = 0;
        while sub_rx.try_recv().is_ok() {
            received += 1;
        }
        assert_eq!(
            received, 1,
            "expected exactly one buffered message in the full subscriber channel"
        );
    }
}
