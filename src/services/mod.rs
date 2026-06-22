///
/// # Neolink Services
///
/// This module handles the controls of the services like http/https/rtmp/rtsp/onvif
/// It only works on newer cameras that have these built in
///
///
/// # Usage
///
/// ```bash
/// # To turn on http
/// neolink services --config=config.toml CameraName http on
/// # Or off
/// neolink services --config=config.toml CameraName http off
/// # Or set the port
/// neolink services --config=config.toml CameraName http port 80
/// # Or get the current port details
/// neolink services --config=config.toml CameraName http get
/// ```
///
/// Services are
/// - http
/// - https
/// - rtmp
/// - rtsp
/// - onvif (will also turn on rtsp)
///
use anyhow::{Context, Result};

mod cmdline;

use crate::common::NeoReactor;
pub(crate) use cmdline::*;

/// Entry point for the pir subcommand
///
/// Opt is the command line options
pub(crate) async fn main(opt: Opt, reactor: NeoReactor) -> Result<()> {
    let camera = reactor.get(&opt.camera).await?;

    match opt.cmd {
        PortAction::Get => match opt.service {
            Services::Http => {
                let state = camera
                    .run_task(|cam| {
                        Box::pin(async move {
                            cam.get_http()
                                .await
                                .context("Unable to get camera service state")
                        })
                    })
                    .await?;
                let ser = String::from_utf8(
                    {
                        let mut buf = bytes::BytesMut::new();
                        quick_xml::se::to_writer(&mut buf, &state).map(|_| buf.to_vec())
                    }
                    .expect("Should Ser the struct"),
                )
                .expect("Should be UTF8");
                println!("{}", ser);
            }
            Services::Https => {
                let state = camera
                    .run_task(|cam| {
                        Box::pin(async move {
                            cam.get_https()
                                .await
                                .context("Unable to get camera service state")
                        })
                    })
                    .await?;
                let ser = String::from_utf8(
                    {
                        let mut buf = bytes::BytesMut::new();
                        quick_xml::se::to_writer(&mut buf, &state).map(|_| buf.to_vec())
                    }
                    .expect("Should Ser the struct"),
                )
                .expect("Should be UTF8");
                println!("{}", ser);
            }
            Services::Rtsp => {
                let state = camera
                    .run_task(|cam| {
                        Box::pin(async move {
                            cam.get_rtsp()
                                .await
                                .context("Unable to get camera service state")
                        })
                    })
                    .await?;
                let ser = String::from_utf8(
                    {
                        let mut buf = bytes::BytesMut::new();
                        quick_xml::se::to_writer(&mut buf, &state).map(|_| buf.to_vec())
                    }
                    .expect("Should Ser the struct"),
                )
                .expect("Should be UTF8");
                println!("{}", ser);
            }
            Services::Rtmp => {
                let state = camera
                    .run_task(|cam| {
                        Box::pin(async move {
                            cam.get_rtmp()
                                .await
                                .context("Unable to get camera service state")
                        })
                    })
                    .await?;
                let ser = String::from_utf8(
                    {
                        let mut buf = bytes::BytesMut::new();
                        quick_xml::se::to_writer(&mut buf, &state).map(|_| buf.to_vec())
                    }
                    .expect("Should Ser the struct"),
                )
                .expect("Should be UTF8");
                println!("{}", ser);
            }
            Services::Onvif => {
                let state = camera
                    .run_task(|cam| {
                        Box::pin(async move {
                            cam.get_onvif()
                                .await
                                .context("Unable to get camera service state")
                        })
                    })
                    .await?;
                let ser = String::from_utf8(
                    {
                        let mut buf = bytes::BytesMut::new();
                        quick_xml::se::to_writer(&mut buf, &state).map(|_| buf.to_vec())
                    }
                    .expect("Should Ser the struct"),
                )
                .expect("Should be UTF8");
                println!("{}", ser);
            }
            Services::Baichuan => {
                let state = camera
                    .run_task(|cam| {
                        Box::pin(async move {
                            cam.get_serverport()
                                .await
                                .context("Unable to get camera service state")
                        })
                    })
                    .await?;
                let ser = String::from_utf8(
                    {
                        let mut buf = bytes::BytesMut::new();
                        quick_xml::se::to_writer(&mut buf, &state).map(|_| buf.to_vec())
                    }
                    .expect("Should Ser the struct"),
                )
                .expect("Should be UTF8");
                println!("{}", ser);
            }
        },
        action => {
            let on = match &action {
                PortAction::On => Some(true),
                PortAction::Off => Some(false),
                PortAction::Set { enabled, .. } => Some(*enabled),
                _ => None,
            };
            // CLI ports are validated as u16 (rejecting > 65535); the camera
            // service methods take u32, so widen here.
            let port = match &action {
                PortAction::Port { port } => Some(u32::from(*port)),
                PortAction::Set { port, .. } => Some(u32::from(*port)),
                _ => None,
            };
            match opt.service {
                Services::Http => {
                    camera
                        .run_task(|cam| {
                            Box::pin(async move {
                                cam.set_http(on, port)
                                    .await
                                    .context("Unable to set camera service state")
                            })
                        })
                        .await?;
                }
                Services::Https => {
                    camera
                        .run_task(|cam| {
                            Box::pin(async move {
                                cam.set_https(on, port)
                                    .await
                                    .context("Unable to set camera service state")
                            })
                        })
                        .await?;
                }
                Services::Rtsp => {
                    camera
                        .run_task(|cam| {
                            Box::pin(async move {
                                // Onvif will not work without rtsp
                                if let Some(false) = &on {
                                    cam.set_onvif(Some(false), None)
                                        .await
                                        .context("Unable to set camera service state")?;
                                }
                                cam.set_rtsp(on, port)
                                    .await
                                    .context("Unable to set camera service state")
                            })
                        })
                        .await?;
                }
                Services::Rtmp => {
                    camera
                        .run_task(|cam| {
                            Box::pin(async move {
                                cam.set_rtmp(on, port)
                                    .await
                                    .context("Unable to set camera service state")
                            })
                        })
                        .await?;
                }
                Services::Onvif => {
                    camera
                        .run_task(|cam| {
                            Box::pin(async move {
                                // Onvif will not work without rtsp
                                if let Some(true) = &on {
                                    cam.set_rtsp(Some(true), None)
                                        .await
                                        .context("Unable to set camera service state")?;
                                }
                                cam.set_onvif(on, port)
                                    .await
                                    .context("Unable to set camera service state")
                            })
                        })
                        .await?;
                }
                Services::Baichuan => {
                    camera
                        .run_task(|cam| {
                            Box::pin(async move {
                                cam.set_serverport(on, port)
                                    .await
                                    .context("Unable to set camera service state")
                            })
                        })
                        .await?;
                }
            }
        }
    }

    Ok(())
}
