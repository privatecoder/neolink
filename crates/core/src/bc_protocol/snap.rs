// use futures::{StreamExt, TryStreamExt};

use super::{BcCamera, Error, Result};
use crate::bc::{model::*, xml::*};
use std::time::Duration;

/// Number of times a short (truncated) snapshot is retried before giving up.
const SNAP_MAX_ATTEMPTS: usize = 3;
/// Delay between snapshot retries, giving a transient packet loss time to clear.
const SNAP_RETRY_DELAY: Duration = Duration::from_millis(500);

/// Decide whether an assembled snapshot is complete.
///
/// The camera declares the JPEG size up front (`expected`). A lost binary chunk
/// during the transfer (e.g. a UDP gap-skip) leaves us with fewer bytes than
/// declared, producing a truncated image that still decodes to only a fraction
/// of the frame. Treat any short result as an error so the caller can retry
/// rather than publish a partial preview. Extra trailing bytes (padding) are
/// tolerated.
fn check_snapshot_complete(actual: usize, expected: usize) -> Result<()> {
    if actual < expected {
        Err(Error::IncompleteSnapshot { expected, actual })
    } else {
        Ok(())
    }
}

impl BcCamera {
    /// Get the snapshot image
    ///
    /// A snapshot is assembled from binary chunks sent by the camera; if one is
    /// lost mid-transfer the result is a truncated JPEG. When that happens the
    /// whole snap is retried a few times before returning
    /// [`Error::IncompleteSnapshot`], so callers never receive a partial image.
    pub async fn get_snapshot(&self) -> Result<Vec<u8>> {
        let mut last_err = None;
        for attempt in 0..SNAP_MAX_ATTEMPTS {
            if attempt > 0 {
                log::debug!(
                    "Retrying snapshot (attempt {} of {})",
                    attempt + 1,
                    SNAP_MAX_ATTEMPTS
                );
                tokio::time::sleep(SNAP_RETRY_DELAY).await;
            }
            match self.get_snapshot_once().await {
                Err(e @ Error::IncompleteSnapshot { .. }) => {
                    log::debug!("{}", e);
                    last_err = Some(e);
                    continue;
                }
                other => return other,
            }
        }
        // All attempts came back short.
        Err(last_err.expect("SNAP_MAX_ATTEMPTS is non-zero so a result was recorded"))
    }

    /// Perform a single snapshot fetch. Returns [`Error::IncompleteSnapshot`] if
    /// the assembled image is shorter than the camera-declared size.
    async fn get_snapshot_once(&self) -> Result<Vec<u8>> {
        let connection = self.get_connection();
        let msg_num = self.new_message_num();
        let mut sub_get = connection.subscribe(MSG_ID_SNAP, msg_num).await?;
        let get = Bc {
            meta: BcMeta {
                msg_id: MSG_ID_SNAP,
                channel_id: self.channel_id,
                msg_num,
                response_code: 0,
                stream_type: 0,
                class: 0x6414,
            },
            body: BcBody::ModernMsg(ModernMsg {
                extension: Some(Extension {
                    channel_id: Some(self.channel_id),
                    ..Default::default()
                }),
                payload: Some(BcPayloads::BcXml(BcXml {
                    snap: Some(Snap {
                        version: "1.1".to_string(),
                        channel_id: self.channel_id,
                        logic_channel: Some(self.channel_id),
                        time: 0,
                        full_frame: Some(0),
                        stream_type: Some("main".to_string()),
                        ..Default::default()
                    }),
                    ..Default::default()
                })),
            }),
        };

        sub_get.send(get).await?;
        let msg = sub_get.recv().await?;
        if msg.meta.response_code != 200 {
            return Err(Error::CameraServiceUnavailable {
                id: msg.meta.msg_id,
                code: msg.meta.response_code,
            });
        }

        if let BcBody::ModernMsg(ModernMsg {
            payload:
                Some(BcPayloads::BcXml(BcXml {
                    snap:
                        Some(Snap {
                            file_name: Some(filename),
                            picture_size: Some(expected_size),
                            ..
                        }),
                    ..
                })),
            ..
        }) = msg.body
        {
            drop(sub_get); // Ensure that we are NOT listening on that msgnum as the reply can come on ANY msgnum
            log::trace!("Got snap XML {} with size {}", filename, expected_size);
            // Messages are now sent on ID 109 but not with the same message ID
            // preumably because the camera considers it to be a new message rather
            // than a reply
            //
            // This means we need to listen for the next 109 grab the message num and
            // subscribe to it. This is what `subscribe_to_next` is for
            let mut sub_get = connection.subscribe_to_id(MSG_ID_SNAP).await?;
            let expected_size = expected_size as usize;

            let mut result: Vec<_> = vec![];
            log::trace!("Waiting for packets on {}", msg_num);
            let mut msg = sub_get.recv().await?;

            while msg.meta.response_code == 200 {
                // sends 200 while more is to come
                //       201 when finished

                if let BcBody::ModernMsg(ModernMsg {
                    extension:
                        Some(Extension {
                            binary_data: Some(1),
                            ..
                        }),
                    payload: Some(BcPayloads::Binary(data)),
                }) = msg.body
                {
                    result.extend_from_slice(&data);
                } else {
                    return Err(Error::UnintelligibleReply {
                        reply: std::sync::Arc::new(Box::new(msg)),
                        why: "Expected binary data but got something else",
                    });
                }
                log::trace!(
                    "Got packet size is now {} of {}",
                    result.len(),
                    expected_size
                );
                msg = sub_get.recv().await?;
            }

            if msg.meta.response_code == 201 {
                // 201 means all binary data sent
                if let BcBody::ModernMsg(ModernMsg {
                    extension:
                        Some(Extension {
                            binary_data: Some(1),
                            ..
                        }),
                    payload,
                }) = msg.body
                {
                    if let Some(BcPayloads::Binary(data)) = payload {
                        // Add last data if present (may be zero if preveious packet contained it)
                        result.extend_from_slice(&data);
                    }
                    log::trace!(
                        "Got all packets size is now {} of {}",
                        result.len(),
                        expected_size
                    );
                    check_snapshot_complete(result.len(), expected_size)?;
                } else {
                    return Err(Error::UnintelligibleReply {
                        reply: std::sync::Arc::new(Box::new(msg)),
                        why: "Expected binary data but got something else",
                    });
                }
            } else {
                // anything else is an error
                return Err(Error::CameraServiceUnavailable {
                    id: msg.meta.msg_id,
                    code: msg.meta.response_code,
                });
            }

            // let binary_stream = sub_get.payload_stream();
            // let result: Vec<_>= binary_stream
            //     .map_ok(|i| tokio_stream::iter(i).map(Result::Ok))
            //     .try_flatten()
            //     .take(expected_size)
            //     .try_collect()
            //     .await?;
            log::trace!("Snapshot recieved: {} of {}", result.len(), expected_size);
            Ok(result)
        } else {
            Err(Error::UnintelligibleReply {
                reply: std::sync::Arc::new(Box::new(msg)),
                why: "Expected Snap xml but it was not recieved",
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_when_exact() {
        assert!(check_snapshot_complete(1024, 1024).is_ok());
    }

    #[test]
    fn complete_when_extra_trailing_bytes() {
        // Cameras occasionally append padding; more bytes than declared is not
        // a truncated image, so it is not an error.
        assert!(check_snapshot_complete(1030, 1024).is_ok());
    }

    #[test]
    fn errors_when_short() {
        let err = check_snapshot_complete(512, 1024);
        assert!(matches!(
            err,
            Err(Error::IncompleteSnapshot {
                expected: 1024,
                actual: 512
            })
        ));
    }
}
