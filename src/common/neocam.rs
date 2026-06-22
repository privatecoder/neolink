//! This is the common code for creating a camera instance
//!
//! Features:
//!    Shared stream BC delivery
//!    Common restart code
//!    Clonable interface to share amongst threadsanyhow::anyhow;
use anyhow::Context;
use futures::stream::StreamExt;
use std::sync::Weak;
use tokio::{
    sync::{
        broadcast::Receiver as BroadcastReceiver,
        mpsc::{channel as mpsc, Sender as MpscSender},
        oneshot::{channel as oneshot, Sender as OneshotSender},
        watch::{channel as watch, Receiver as WatchReceiver, Sender as WatchSender},
    },
    task::JoinSet,
    time::{sleep, Duration},
};
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;

use super::{
    DoorbellEvent, MdRequest, MdState, NeoCamMdThread, NeoCamThread, NeoCamThreadState,
    NeoInstance, Permit, UseCounter,
};
use crate::{
    config::{CameraConfig, ConnectMode},
    AnyResult, Result,
};
use neolink_core::bc_protocol::BcCamera;

pub(crate) enum NeoCamCommand {
    HangUp,
    Instance(OneshotSender<Result<NeoInstance>>),
    Motion(OneshotSender<WatchReceiver<MdState>>),
    Doorbell(OneshotSender<BroadcastReceiver<DoorbellEvent>>),
    Config(OneshotSender<WatchReceiver<CameraConfig>>),
    Disconnect(OneshotSender<()>),
    Connect(OneshotSender<()>),
    GetPermit(OneshotSender<Permit>),
}
/// The underlying camera binding
pub(crate) struct NeoCam {
    cancel: CancellationToken,
    config_watch: WatchSender<CameraConfig>,
    commander: MpscSender<NeoCamCommand>,
    camera_watch: WatchReceiver<Weak<BcCamera>>,
    set: JoinSet<AnyResult<()>>,
}

impl NeoCam {
    pub(crate) async fn new(config: CameraConfig) -> Result<NeoCam> {
        let (commander_tx, commander_rx) = mpsc(100);
        let (watch_config_tx, watch_config_rx) = watch(config.clone());
        let (camera_watch_tx, camera_watch_rx) = watch(Weak::new());
        let (md_request_tx, md_request_rx) = mpsc(100);
        // Start disconnected - camera will connect when first RTSP client arrives (via permit system)
        let (state_tx, state_rx) = watch(NeoCamThreadState::Disconnected);

        let set = JoinSet::new();
        let users = UseCounter::new().await;

        let mut me = Self {
            cancel: CancellationToken::new(),
            config_watch: watch_config_tx,
            commander: commander_tx.clone(),
            camera_watch: camera_watch_rx.clone(),
            set,
        };

        // This thread recieves messages from the instances
        // and acts on it.
        //
        // This thread must be started first so that we can begin creating instances for the
        // other threads
        let sender_cancel = me.cancel.clone();
        let mut commander_rx = ReceiverStream::new(commander_rx);
        let thread_commander_tx = commander_tx.clone();
        let thread_watch_config_rx = watch_config_rx.clone();

        me.set.spawn(async move {
            let thread_cancel = sender_cancel.clone();
            let res = tokio::select! {
                _ = sender_cancel.cancelled() => {
                    Result::Ok(())
                },
                v = async {
                    while let Some(command) = commander_rx.next().await {
                        match command {
                            NeoCamCommand::HangUp => {
                                sender_cancel.cancel();
                                return Result::<(), anyhow::Error>::Ok(());
                            }
                            NeoCamCommand::Instance(result) => {
                                let instance = NeoInstance::new(
                                    camera_watch_rx.clone(),
                                    thread_commander_tx.clone(),
                                    thread_cancel.clone(),
                                );
                                let _ = result.send(instance);
                            }
                            NeoCamCommand::Motion(sender) => {
                                md_request_tx.send(
                                    MdRequest::Get {
                                        sender,
                                    }
                                ).await?;
                            },
                            NeoCamCommand::Doorbell(sender) => {
                                md_request_tx.send(
                                    MdRequest::GetDoorbell {
                                        sender,
                                    }
                                ).await?;
                            },
                            NeoCamCommand::Config(sender) => {
                                let _ = sender.send(thread_watch_config_rx.clone());
                            },
                            NeoCamCommand::Connect(sender) => {
                                if !matches!(*state_tx.borrow(), NeoCamThreadState::Connected) {
                                    state_tx.send_replace(NeoCamThreadState::Connected);
                                }
                                let _ = sender.send(());
                            }
                            NeoCamCommand::Disconnect(sender) => {
                                if !matches!(*state_tx.borrow(), NeoCamThreadState::Disconnected) {
                                    state_tx.send_replace(NeoCamThreadState::Disconnected);
                                }
                                let _ = sender.send(());
                            }
                            NeoCamCommand::GetPermit(sender) => {
                                let _ = sender.send(users.create_activated().await?);
                            }
                        }
                    }
                    Ok(())
                } => {
                    v
                }
            };
            res
        });

        // This gets the first instance which we use for making the other threads
        let (instance_tx, instance_rx) = oneshot();
        commander_tx
            .send(NeoCamCommand::Instance(instance_tx))
            .await?;
        let instance = instance_rx.await??;

        // This thread maintains the camera loop
        //
        // It will keep it logged and reconnect
        let thread_watch_config_rx = watch_config_rx.clone();
        let mut cam_thread = NeoCamThread::new(
            state_rx,
            thread_watch_config_rx,
            camera_watch_tx,
            me.cancel.clone(),
        )
        .await;
        me.set.spawn(async move { cam_thread.run().await });

        // This thread monitors the motion
        let md_instance = instance.subscribe().await?;
        let md_cancel = me.cancel.clone();
        let mut md_thread = NeoCamMdThread::new(md_request_rx, md_instance).await?;
        me.set.spawn(async move {
            tokio::select! {
                _ = md_cancel.cancelled() => AnyResult::Ok(()),
                v = md_thread.run() => {
                    v
                },
            }
        });

        // Camera info reporting removed - was triggering connection at startup
        // Model/firmware info and UID queries now happen only when camera is already
        // connected for another reason (RTSP client, MQTT command, etc.)
        let _report_instance = instance.subscribe().await?;
        let _uid_instance = instance.clone();

        // MD permits
        let md_permit_instance = instance.subscribe().await?;
        let md_permit_cancel = me.cancel.clone();
        me.set.spawn(async move {
            tokio::select! {
                _ = md_permit_cancel.cancelled() => {
                    AnyResult::Ok(())
                },
                v = async {
                    let mut md = md_permit_instance.motion().await.with_context(|| "Unable to acquire motion watcher")?;
                    loop{
                        md.wait_for(|md| matches!(md, MdState::Start(_))).await.with_context(|| "MD Watcher lost")?;
                        let _permit = md_permit_instance.permit().await.with_context(|| "Unuable to acquire motion permit")?;
                        md.wait_for(|md| matches!(md, MdState::Stop(_))).await.with_context(|| "MD Watcher lost")?;
                        // Try waiting for 30s
                        // If in those 30s we get motion then return to
                        // loop early to reaquire the permit
                        tokio::select!{
                            _ = sleep(Duration::from_secs(30)) => {},
                            v = md.wait_for(|md| matches!(md, MdState::Start(_))) => {v.with_context(|| "MD Watcher lost")?;},
                        }
                    }
                } => {
                    v
                },
            }
        });

        // This thread manages the camera connection lifecycle per `connect_mode`:
        //   - Always:    connect at startup and stay connected; if idle_timeout_secs > 0,
        //                disconnect after that long idle and reconnect on demand.
        //   - OnDemand:  connect only when a permit is held; disconnect when idle, after
        //                an optional relay_warm_seconds grace.
        // Permits are created when RTSP clients connect or when other tasks need the camera.
        // The loop re-reads the config each cycle so changes take effect.
        let connect_instance = instance.subscribe().await?;
        let connect_cancel = me.cancel.clone();
        me.set.spawn(async move {
            tokio::select!{
                _ = connect_cancel.cancelled() => {
                    AnyResult::Ok(())
                },
                v = async {
                    let mut permit = connect_instance.permit().await?;
                    permit.deactivate().await?; // Watching only, don't count as active

                    let mut config_rx = connect_instance.config().await?;
                    let name = config_rx.borrow().name.clone();

                    'lifecycle: loop {
                        // Snapshot the mode (and mark the current config as seen so a
                        // later `changed()` only fires on a genuine config change).
                        let mode = config_rx.borrow_and_update().connect_mode;
                        match mode {
                            ConnectMode::Always => {
                                // Connect at startup / stay connected (idempotent).
                                connect_instance.connect().await?;
                                let idle_timeout = config_rx.borrow().idle_timeout_secs;
                                if idle_timeout == 0 {
                                    // Never idle-disconnect; just wait for a config change.
                                    if config_rx.changed().await.is_err() {
                                        break;
                                    }
                                    continue;
                                }
                                // idle_timeout > 0: disconnect after being idle that long.
                                tokio::select! {
                                    r = config_rx.changed() => { if r.is_err() { break; } continue; }
                                    r = permit.dropped_users() => {
                                        r?;
                                        let idle_deadline = sleep(Duration::from_secs(idle_timeout));
                                        tokio::pin!(idle_deadline);
                                        tokio::select! {
                                            _ = &mut idle_deadline => {
                                                log::info!("{name}: idle {idle_timeout}s, disconnecting (connect_mode=always)");
                                                connect_instance.disconnect().await?;
                                                // Reconnect on demand (or re-evaluate on config change).
                                                tokio::select! {
                                                    r = permit.aquired_users() => { r?; }
                                                    r = config_rx.changed() => { if r.is_err() { break; } }
                                                }
                                            }
                                            // Re-used before timeout: stay connected.
                                            r = permit.aquired_users() => { r?; }
                                            r = config_rx.changed() => { if r.is_err() { break; } }
                                        }
                                        continue;
                                    }
                                }
                            }
                            ConnectMode::OnDemand => {
                                // Wait for demand before connecting.
                                tokio::select! {
                                    r = config_rx.changed() => { if r.is_err() { break; } continue; }
                                    r = permit.aquired_users() => { r?; }
                                }
                                log::info!("{name}: Permit acquired, connecting to camera relay");
                                connect_instance.connect().await?;

                                // Stay connected until all permits drop, plus a warm grace.
                                loop {
                                    // Wait for users to drop, but also react to config
                                    // changes (e.g. connect_mode) while connected. Users
                                    // are present here, so re-evaluating re-reads the mode
                                    // without dropping the active stream (connect() is
                                    // idempotent and aquired_users() returns immediately).
                                    tokio::select! {
                                        r = permit.dropped_users() => { r?; }
                                        r = config_rx.changed() => {
                                            if r.is_err() { break 'lifecycle; }
                                            log::info!("{name}: config changed; re-evaluating connect mode");
                                            continue 'lifecycle;
                                        }
                                    }
                                    let warm_secs = config_rx.borrow().relay_warm_seconds;
                                    if warm_secs == 0 {
                                        break;
                                    }
                                    log::info!(
                                        "{name}: All permits dropped, keeping relay warm for {}s",
                                        warm_secs
                                    );
                                    let warm_deadline = sleep(Duration::from_secs(warm_secs));
                                    tokio::pin!(warm_deadline);
                                    tokio::select! {
                                        _ = &mut warm_deadline => {
                                            break;
                                        }
                                        v = permit.aquired_users() => {
                                            v?;
                                            log::info!("{name}: Permit reacquired during warm relay window");
                                            continue;
                                        }
                                        r = config_rx.changed() => {
                                            if r.is_err() { break 'lifecycle; }
                                            // No active users during the warm window, so
                                            // ending it (disconnect) to re-evaluate the mode
                                            // is safe and disrupts nothing.
                                            log::info!("{name}: config changed during warm window; re-evaluating");
                                            break;
                                        }
                                    }
                                }

                                log::info!("{name}: All permits dropped, disconnecting from camera relay");
                                connect_instance.disconnect().await?;
                            }
                        }
                    }
                    AnyResult::Ok(())
                } => {
                    v
                },
            }
        });

        Ok(me)
    }

    /// Whether this camera's task set is still live. A `NeoCam` cancels its
    /// shared `cancel` token when its tasks stop (notably on a fatal
    /// `CameraLoginFail`, where every task is torn down). Once cancelled the
    /// instance is a zombie — its command channel is dead — so the reactor uses
    /// this to decide whether to recreate it instead of handing it back.
    pub(crate) fn is_alive(&self) -> bool {
        !self.cancel.is_cancelled()
    }

    pub(crate) async fn subscribe(&self) -> Result<NeoInstance> {
        NeoInstance::new(
            self.camera_watch.clone(),
            self.commander.clone(),
            self.cancel.clone(),
        )
    }

    pub(crate) async fn update_config(&self, config: CameraConfig) -> Result<()> {
        self.config_watch.send_replace(config);
        Ok(())
    }
}

impl Drop for NeoCam {
    fn drop(&mut self) {
        log::trace!("Drop NeoCam");
        let mut set = std::mem::take(&mut self.set);
        let commander = self.commander.clone();
        let _gt = tokio::runtime::Handle::current().enter();
        tokio::task::spawn(async move {
            let _ = commander.send(NeoCamCommand::HangUp).await;
            while set.join_next().await.is_some() {}
            log::trace!("Dropped NeoCam");
        });
    }
}
