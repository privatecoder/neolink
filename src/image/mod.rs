///
/// # Neolink Image
///
/// This module can be used to dump a still image from the camera
///
///
/// # Usage
/// ```bash
/// neolink image --config=config.toml --file-path=filepath CameraName
/// ```
///
/// Cameras that do not support the SNAP command need to use `--use_stream`
/// which will make the camera play the stream and transcode the video into a jpeg
/// e.g.:
///
/// ```bash
/// neolink image --config=config.toml --use_stream --file-path=filepath CameraName
/// ```
///
use anyhow::{anyhow, Result};
use log::*;
use neolink_core::{
    bc_protocol::*,
    bcmedia::model::{BcMedia, BcMediaIframe, BcMediaPframe},
};
use std::sync::Arc;
use tokio::{fs::File, io::AsyncWriteExt, sync::RwLock};

mod cmdline;
mod gst;

use crate::common::NeoReactor;
pub(crate) use cmdline::Opt;

/// Entry point for the image subcommand
///
/// Opt is the command line options
pub(crate) async fn main(opt: Opt, reactor: NeoReactor) -> Result<()> {
    let camera = reactor.get(&opt.camera).await?;

    if opt.use_stream {
        let (stream_data_tx, mut stream_data_rx) = tokio::sync::mpsc::channel(500);

        // Spawn a video stream
        let thread_camera = camera.clone();
        let (stream_type_tx, stream_type_rx) = tokio::sync::oneshot::channel();
        let stream_type_tx = Arc::new(RwLock::new(Some(stream_type_tx)));
        tokio::task::spawn(async move {
            thread_camera
                .run_task(|cam| {
                    let stream_type_tx = stream_type_tx.clone();
                    let stream_data_tx = stream_data_tx.clone();

                    Box::pin(async move {
                        let mut stream = cam.start_video(StreamKind::Main, 500, false).await?;
                        while let Ok(frame) = stream.get_data().await {
                            let frame = frame?;
                            match frame {
                                BcMedia::Iframe(BcMediaIframe {
                                    data, video_type, ..
                                })
                                | BcMedia::Pframe(BcMediaPframe {
                                    data, video_type, ..
                                }) => {
                                    if let Some(stream_type_tx) =
                                        stream_type_tx.write().await.take()
                                    {
                                        let _ = stream_type_tx.send(video_type);
                                    }
                                    stream_data_tx.send(Arc::new(data)).await?;
                                }
                                _ => {}
                            }
                        }
                        Result::Ok(())
                    })
                })
                .await
        });

        let vid_type = stream_type_rx.await?;
        let buf = stream_data_rx
            .recv()
            .await
            .ok_or(anyhow!("No frames recieved"))?;

        let mut sender = gst::from_input(vid_type, &opt.file_path).await?;
        sender.send(buf).await?; // Send first iframe

        // Keep sending both IFrame or PFrame until finished
        while sender.is_finished().await.is_none() {
            if let Some(buf) = stream_data_rx.recv().await {
                debug!("Sending frame data to gstreamer");
                if sender.send(buf).await.is_err() {
                    // Assume that the sender is closed
                    // because the pipeline is finished
                    break;
                }
            } else {
                log::error!("Camera stopped sending frames before decoding could complete");
                break;
            }
        }
        debug!("Sending EOS");
        let _ = sender.eos().await; // Ignore return because if pipeline is finished this will error
        let _ = sender.join().await;
    } else {
        // Simply use the snap command
        debug!("Using the snap command");
        let file_path = opt.file_path.with_extension("jpeg");
        let mut buffer = File::create(file_path).await?;
        let jpeg_data = camera
            .run_task(|camera| Box::pin(async move { Ok(camera.get_snapshot().await?) }))
            .await;
        if jpeg_data.is_err() {
            log::debug!("jpeg_data: {:?}", jpeg_data);
        }
        let jpeg_data = jpeg_data?;
        buffer.write_all(jpeg_data.as_slice()).await?;
    }

    Ok(())
}
