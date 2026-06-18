// use futures::{StreamExt, TryStreamExt};

use super::{BcCamera, Error, Result};
use crate::bc::{model::*, xml::*};
use std::time::Duration;

/// Number of times a short (truncated) snapshot is retried before giving up.
const SNAP_MAX_ATTEMPTS: usize = 3;
/// Delay between snapshot retries, giving a transient packet loss time to clear.
const SNAP_RETRY_DELAY: Duration = Duration::from_millis(500);

/// JPEG Start-Of-Image marker (the first two bytes of every JPEG).
const JPEG_SOI: [u8; 2] = [0xFF, 0xD8];
/// JPEG End-Of-Image marker (the last two bytes of a complete JPEG).
const JPEG_EOI: [u8; 2] = [0xFF, 0xD9];

/// Validate an assembled snapshot as a structurally complete JPEG and return
/// the byte length up to and including the last EOI marker.
///
/// The camera's declared `picture_size` is untrustworthy: for a truncated
/// snapshot it declares the truncated length, so a byte-count check passes even
/// though the image is cut off (it decodes to only the top fraction of the
/// frame). Judge completeness from the JPEG framing instead: the data must be
/// non-empty, start with the SOI marker (`FF D8`), and contain an EOI marker
/// (`FF D9`) after the SOI.
///
/// The returned length stops at the last EOI; any bytes beyond it are
/// camera/transport padding and are excluded, so the caller can truncate the
/// assembled buffer to exactly the JPEG. Anything without a valid SOI+EOI is
/// treated as incomplete so the caller can retry rather than publish a partial
/// preview.
fn check_snapshot_complete(data: &[u8]) -> Result<usize> {
    if data.is_empty() {
        return Err(Error::IncompleteSnapshot {
            actual: 0,
            reason: "empty snapshot",
        });
    }
    if !data.starts_with(&JPEG_SOI) {
        return Err(Error::IncompleteSnapshot {
            actual: data.len(),
            reason: "missing SOI marker",
        });
    }
    // The image ends at the last EOI marker that occurs after the SOI; trailing
    // bytes past it are padding. `skip(JPEG_SOI.len())` keeps the search after
    // the SOI, and `.next_back()` selects the final EOI by scanning from the end.
    match data
        .windows(JPEG_EOI.len())
        .enumerate()
        .skip(JPEG_SOI.len())
        .filter(|(_, window)| *window == JPEG_EOI)
        .map(|(start, _)| start + JPEG_EOI.len())
        .next_back()
    {
        Some(end) => Ok(end),
        None => Err(Error::IncompleteSnapshot {
            actual: data.len(),
            reason: "missing EOI marker",
        }),
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
        // Every attempt came back incomplete (truncated JPEG).
        Err(last_err.expect("SNAP_MAX_ATTEMPTS is non-zero so a result was recorded"))
    }

    /// Perform a single snapshot fetch. Returns [`Error::IncompleteSnapshot`] if
    /// the assembled image is not a structurally complete JPEG (missing
    /// SOI/EOI markers).
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
                        // Request the complete full-resolution frame. With
                        // full_frame=0 the camera returns a truncated ~15 KB
                        // preview slice (no EOI marker) yet declares that
                        // truncated length as picture_size, so it cannot be
                        // detected by byte count alone.
                        full_frame: Some(1),
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
            // Verify the assembled bytes are a complete JPEG and trim any
            // trailing padding after the final EOI. The camera's declared
            // `expected_size` is not trusted for this (it matches even a
            // truncated image); completeness is judged from the JPEG markers.
            let valid_len = check_snapshot_complete(&result)?;
            result.truncate(valid_len);
            log::trace!(
                "Snapshot recieved: {} bytes (declared {})",
                result.len(),
                expected_size
            );
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
    fn accepts_minimal_valid_jpeg() {
        // SOI ... EOI, the smallest structurally complete JPEG framing.
        let jpeg = [0xFF, 0xD8, 0xFF, 0xD9];
        // Valid end index is the whole buffer (ends exactly at EOI).
        assert_eq!(check_snapshot_complete(&jpeg).unwrap(), 4);
    }

    #[test]
    fn accepts_jpeg_with_body() {
        let jpeg = [0xFF, 0xD8, 0x12, 0x34, 0x56, 0xFF, 0xD9];
        assert_eq!(check_snapshot_complete(&jpeg).unwrap(), 7);
    }

    #[test]
    fn accepts_jpeg_with_trailing_padding() {
        // A complete JPEG followed by camera/transport padding after the EOI.
        // The returned length must point at the end of the EOI so the caller can
        // strip the padding; the truncated bytes end exactly at FF D9.
        let jpeg = [0xFF, 0xD8, 0x12, 0x34, 0x56, 0xFF, 0xD9, 0x00, 0x00];
        let end = check_snapshot_complete(&jpeg).unwrap();
        assert_eq!(end, 7);
        assert_eq!(&jpeg[end - 2..end], &JPEG_EOI);
        assert_eq!(&jpeg[..end], &[0xFF, 0xD8, 0x12, 0x34, 0x56, 0xFF, 0xD9]);
    }

    #[test]
    fn accepts_last_eoi_when_multiple() {
        // If more than one EOI is present, truncate at the last one.
        let jpeg = [0xFF, 0xD8, 0xFF, 0xD9, 0x11, 0xFF, 0xD9, 0x22];
        assert_eq!(check_snapshot_complete(&jpeg).unwrap(), 7);
    }

    #[test]
    fn rejects_truncated_missing_eoi() {
        // The real-world failure: a valid SOI and body but no EOI marker, which
        // the camera nonetheless declared as the full picture_size.
        let jpeg = [0xFF, 0xD8, 0x12, 0x34, 0x56];
        assert!(matches!(
            check_snapshot_complete(&jpeg),
            Err(Error::IncompleteSnapshot {
                actual: 5,
                reason: "missing EOI marker",
            })
        ));
    }

    #[test]
    fn rejects_empty() {
        assert!(matches!(
            check_snapshot_complete(&[]),
            Err(Error::IncompleteSnapshot {
                actual: 0,
                reason: "empty snapshot",
            })
        ));
    }

    #[test]
    fn rejects_non_jpeg_missing_soi() {
        let not_jpeg = [0x00, 0x01, 0x02, 0xFF, 0xD9];
        assert!(matches!(
            check_snapshot_complete(&not_jpeg),
            Err(Error::IncompleteSnapshot {
                actual: 5,
                reason: "missing SOI marker",
            })
        ));
    }
}
