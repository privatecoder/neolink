use anyhow::anyhow;
use gstreamer_rtsp_server::prelude::*;
use std::collections::HashSet;

use crate::{common::NeoInstance, AnyResult};
use neolink_core::bc_protocol::StreamKind;

use super::{factory::*, gst::NeoRtspServer};

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
    // Create the factory
    let (factory, thread) = make_factory(camera, stream).await?;

    factory.add_permitted_roles(users);

    for path in paths.iter() {
        log::info!("{}: Mounting factory at path: {}", name, path);
        mounts.add_factory(path, factory.clone());
    }
    log::info!("{}: Available at {}", name, paths.join(", "));

    thread.await??;
    Ok(())
}
