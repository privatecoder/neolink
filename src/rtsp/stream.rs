use anyhow::anyhow;
use gstreamer_rtsp_server::prelude::*;
use std::collections::HashSet;

use crate::{common::NeoInstance, AnyResult};
use neolink_core::bc_protocol::StreamKind;
use tokio_util::sync::CancellationToken;

use super::{factory::*, gst::NeoRtspServer};

/// Removes the mounted factory paths when this stream generation ends (returns,
/// errors, or is dropped on a config-change restart), so old mounts don't leak.
struct UnmountGuard<'a> {
    mounts: &'a gstreamer_rtsp_server::RTSPMountPoints,
    paths: &'a [String],
    name: &'a str,
}

impl Drop for UnmountGuard<'_> {
    fn drop(&mut self) {
        for path in self.paths {
            log::info!("{}: Unmounting factory from path: {}", self.name, path);
            self.mounts.remove_factory(path);
        }
    }
}

/// This handles the stream itself by creating the factory and pushing messages into it
pub(crate) async fn stream_main(
    camera: NeoInstance,
    stream: StreamKind,
    rtsp: &NeoRtspServer,
    users: &HashSet<String>,
    paths: &[String],
) -> AnyResult<()> {
    let name = camera.config().await?.borrow().name.clone();
    // Create the factory and connect the stream
    let mounts = rtsp
        .mount_points()
        .ok_or(anyhow!("RTSP server lacks mount point"))?;

    // Remove dummy factories from all paths first
    for path in paths.iter() {
        log::info!("{}: Removing dummy factory from path: {}", name, path);
        mounts.remove_factory(path);
    }

    // Cancel the factory's message-handler task when this generation ends/drops
    // (e.g. a config-change restart), instead of leaking it.
    let cancel = CancellationToken::new();
    let _cancel_guard = cancel.clone().drop_guard();

    // Create the real factory with permit-based camera connection
    let (factory, thread) = make_factory(camera, stream, cancel).await?;

    factory.add_permitted_roles(users);

    for path in paths.iter() {
        log::info!("{}: Mounting real factory at path: {}", name, path);
        mounts.add_factory(path, factory.clone());
    }
    // Unmount these paths on teardown so a restarted/disabled stream doesn't leak.
    let _unmount = UnmountGuard {
        mounts: &mounts,
        paths,
        name: &name,
    };
    log::info!("{}: Available at {}", name, paths.join(", "));

    thread.await??;
    Ok(())
}
