//! This thread will listen to motion messages
//! from the camera.

use anyhow::Context;
use std::sync::Arc;
use tokio::{
    sync::{
        broadcast::{
            channel as broadcast, error::RecvError as BroadcastRecvError,
            Receiver as BroadcastReceiver, Sender as BroadcastSender,
        },
        mpsc::Receiver as MpscReceiver,
        oneshot::Sender as OneshotSender,
        watch::{channel as watch, Receiver as WatchReceiver, Sender as WatchSender},
    },
    time::{sleep, Duration, Instant},
};
use tokio_util::sync::CancellationToken;

use super::NeoInstance;
use crate::{AnyResult, Result};
use neolink_core::bc_protocol::MotionStatus;

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub(crate) enum MdState {
    Start(Instant),
    Stop(Instant),
    Unknown,
}

/// A discrete doorbell ("visitor") press.
///
/// Unlike [`MdState`], doorbell presses are events rather than durable state, so
/// they are delivered over a broadcast channel and never modelled as a watch.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DoorbellEvent {
    /// When the press was observed
    #[allow(dead_code)]
    pub(crate) when: Instant,
}

pub(crate) struct NeoCamMdThread {
    md_watcher: Arc<WatchSender<MdState>>,
    doorbell_sender: Arc<BroadcastSender<DoorbellEvent>>,
    md_request_rx: MpscReceiver<MdRequest>,
    cancel: CancellationToken,
    instance: NeoInstance,
}

impl NeoCamMdThread {
    pub(crate) async fn new(
        md_request_rx: MpscReceiver<MdRequest>,
        instance: NeoInstance,
    ) -> Result<Self> {
        let (md_watcher, _) = watch(MdState::Unknown);
        let md_watcher = Arc::new(md_watcher);
        let (doorbell_sender, _) = broadcast(50);
        let doorbell_sender = Arc::new(doorbell_sender);
        Ok(Self {
            md_watcher,
            doorbell_sender,
            md_request_rx,
            cancel: CancellationToken::new(),
            instance,
        })
    }

    pub(crate) async fn run(&mut self) -> Result<()> {
        let thread_cancel = self.cancel.clone();
        let watcher = self.md_watcher.clone();
        let doorbell_sender = self.doorbell_sender.clone();
        let md_instance = self.instance.clone();
        tokio::select! {
            _ = thread_cancel.cancelled() => {
                Ok(())
            },
            v = async {
                while let Some(request) = self.md_request_rx.recv().await {
                    match request {
                        MdRequest::Get {
                            sender
                        } => {
                          let _ = sender.send(self.md_watcher.subscribe());
                        },
                        MdRequest::GetDoorbell {
                            sender
                        } => {
                          let _ = sender.send(self.doorbell_sender.subscribe());
                        },
                    }
                }
                Ok(())
            } => v,
            v = async {
                loop {
                    let r: AnyResult<()> = md_instance.run_passive_task(|cam| {
                        let watcher = watcher.clone();
                        let doorbell_sender = doorbell_sender.clone();
                        Box::pin(
                        async move {
                            let mut md = cam.listen_on_motion().await.with_context(|| "Error in getting MD listen_on_motion")?;
                            // Doorbell presses ride the same camera subscription:
                            // they are decoded from the alarm stream alongside
                            // motion, so there is no second subscription here.
                            let mut doorbell = md.doorbell();
                            loop {
                                tokio::select! {
                                    event = md.next_motion() => {
                                        let event = event.with_context(|| "Error in getting MD next_motion")?;
                                        match event {
                                            MotionStatus::Start(at) => {
                                                watcher.send_replace(
                                                    MdState::Start(at.into())
                                                );
                                            }
                                            MotionStatus::Stop(at) => {
                                                watcher.send_replace(
                                                    MdState::Stop(at.into())
                                                );
                                            }
                                            MotionStatus::NoChange(_) => {},
                                        }
                                    }
                                    press = doorbell.recv() => {
                                        match press {
                                            Ok(when) => {
                                                // No subscribers is fine; the
                                                // press is simply dropped.
                                                let _ = doorbell_sender.send(DoorbellEvent { when: when.into() });
                                            }
                                            // Rare presses mean lagging is
                                            // unlikely; if it happens, skip on.
                                            Err(BroadcastRecvError::Lagged(_)) => {},
                                            // The motion stream ended; surface it
                                            // so the task restarts like next_motion.
                                            Err(BroadcastRecvError::Closed) => {
                                                return Err(anyhow::anyhow!("Doorbell stream closed"));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    )}).await;
                    log::debug!("Error in MD task Restarting: {:?}", r);
                    sleep(Duration::from_secs(1)).await;
                }
            } => v
        }
    }
}

impl Drop for NeoCamMdThread {
    fn drop(&mut self) {
        log::trace!("Drop NeoCamMdThread");
        self.cancel.cancel();
        log::trace!("Dropped NeoCamMdThread");
    }
}

/// Used to pass messages to the MdThread
pub(crate) enum MdRequest {
    Get {
        sender: OneshotSender<WatchReceiver<MdState>>,
    },
    GetDoorbell {
        sender: OneshotSender<BroadcastReceiver<DoorbellEvent>>,
    },
}
