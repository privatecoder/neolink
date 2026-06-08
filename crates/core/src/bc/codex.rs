//! Create a tokio encoder/decoder for turning a AsyncRead/Write stream into
//! a Bc packet
//!
//! BcCodex is used with a `[tokio_util::codec::Framed]` to form complete packets
//!
use crate::bc::model::*;
use crate::bc::xml::*;
use crate::{Credentials, Error, Result};
use bytes::{Buf, BytesMut};
use tokio_util::codec::{Decoder, Encoder};

pub(crate) struct BcCodex {
    context: BcContext,
    /// Bytes discarded while resyncing the stream after a desync (e.g. a UDP
    /// packet was lost/skipped, punching a hole in the framing). Reset to 0
    /// once a message decodes cleanly again.
    amount_skipped: usize,
}

impl BcCodex {
    pub(crate) fn new_with_debug(credentials: Credentials) -> Self {
        let mut context = BcContext::new(credentials);

        context.debug_on();
        Self {
            context,
            amount_skipped: 0,
        }
    }
    pub(crate) fn new(credentials: Credentials) -> Self {
        Self {
            context: BcContext::new(credentials),
            amount_skipped: 0,
        }
    }
}

/// Find the byte offset of the next BC header magic, searching from offset 1 so
/// a resync always makes forward progress — even when `Bc::deserialize` failed
/// *after* a valid magic (e.g. a truncated/garbled body), in which case a valid
/// magic still sits at offset 0 and must be skipped past.
fn next_bc_magic_offset(src: &[u8]) -> Option<usize> {
    if src.len() < 5 {
        return None;
    }
    src.windows(4)
        .enumerate()
        .skip(1)
        .find(|(_, w)| {
            let magic = u32::from_le_bytes([w[0], w[1], w[2], w[3]]);
            magic == MAGIC_HEADER || magic == MAGIC_HEADER_REV
        })
        .map(|(offset, _)| offset)
}

impl Encoder<Bc> for BcCodex {
    type Error = Error;

    fn encode(&mut self, item: Bc, dst: &mut BytesMut) -> Result<()> {
        // let context = self.context.read().unwrap();
        const BC_ENCRYPTED: EncryptionProtocol = EncryptionProtocol::BCEncrypt;
        let buf: Vec<u8> = Default::default();
        let enc_protocol: &EncryptionProtocol = match self.context.get_encrypted() {
            EncryptionProtocol::Aes { .. } | EncryptionProtocol::FullAes { .. }
                if item.meta.msg_id == 1 =>
            {
                // During login the encyption protocol cannot go higher than BCEncrypt
                // even if we support AES. (BUt it can go lower i.e. None)
                &BC_ENCRYPTED
            }
            n => n,
        };
        let buf = item.serialize(buf, enc_protocol)?;
        dst.extend_from_slice(buf.as_slice());
        Ok(())
    }
}

impl Decoder for BcCodex {
    type Item = Bc;
    type Error = Error;

    fn decode_eof(&mut self, buf: &mut BytesMut) -> Result<Option<Self::Item>> {
        match self.decode(buf)? {
            Some(frame) => Ok(Some(frame)),
            None => {
                if buf.is_empty() {
                    Ok(None)
                } else {
                    log::debug!("bytes remaining on BC stream: {:X?}", buf.chunks(25).next());
                    // Right after this we seem to get an issue with the camera dropping us
                    // Needs probing
                    // F0, DE, BC, A, 3, 0, 0, 0, 88, 6, 0, 0, 0, 1, 4, 0, C8, 0, 0, 0, 0, 0, 0, 0, 30, 31, 64, 63, 48,
                    // 32, 36, 34, 6A, 6, 0, 0, 0, 0, 0, 0, D8, F5, C7, 86, 56, 0, 0, 0, 0, 0, 0, 1, 21, 9A, FC, 22, 7F, 6, AE, F6, 15, FF, E5, 71, 4, 2F, 24, 61, 15, 96, F0, BF, 83, DE, 10, BE, B4, 2E, 3
                    // 9, 76, 56, 92, 7E, 48, 79, 20, 9A, DC, 1B, BB, AC, 22, 60, 5C, 72, B5, 3D, 8, E0, 34, 43, 3F, 2E, A7, 81, A8, 11, 75, 7F, 58, 3E, 8, 54, 91, 43, 21, EC, 6B, D6, 1A, D5, CB, D5, 6C,
                    // 8C, 2E, 6E, A3, 51, C3, A4, F0, CF, 2B, 61, 81, D0, 1C, A1, 76, EE, BF, 7A, D5, D8, D1, C4, D, B0, 45, EE, 3E, 93, 9A, CE, 5F, AB, 75, 55, AC, 9D, 66, DE, 23, 6D, 5F, 25, 57, DA, F5
                    //, E, 7F, 8D, 30, A7, 66, C4, 60, 76, 41, D0, 6A, 23, E, A9, C5, 51, EE, F6, DD, 19, E7, A8, 96, 9F, 2B, AF, 31, 90, 9D, FC, BE
                    Ok(None)
                }
            }
        }
        // match self.decode(buf)? {
        //     Some(frame) => Ok(Some(frame)),
        //     None => Ok(None),
        // }
    }

    fn decode(&mut self, src: &mut BytesMut) -> Result<Option<Self::Item>> {
        // trace!("Decoding: {:X?}", src);
        let bc = loop {
            match Bc::deserialize(&self.context, src) {
                Ok(bc) => {
                    if self.amount_skipped > 0 {
                        log::debug!(
                            "BC stream resynced after skipping {} byte(s)",
                            self.amount_skipped
                        );
                        self.amount_skipped = 0;
                    }
                    break bc;
                }
                Err(Error::NomIncomplete(_)) => return Ok(None),
                Err(e) => {
                    // A lost/skipped UDP packet punches a hole in the byte stream
                    // that desyncs the BC framing. Rather than tearing the whole
                    // connection down (which forces a reconnect + re-login), scan
                    // forward to the next BC header magic and resume. BC headers
                    // are plaintext and length-prefixed, and each message decrypts
                    // independently, so resyncing is safe and costs only the one
                    // corrupted message.
                    if self.amount_skipped == 0 {
                        log::debug!("BC stream desync ({e:?}); resyncing to next header magic");
                    }
                    match next_bc_magic_offset(src) {
                        Some(offset) => {
                            self.amount_skipped += offset;
                            src.advance(offset);
                            // Retry deserialize from the recovered position.
                        }
                        None => {
                            // No magic visible yet. Drop everything but a short
                            // tail (a magic may straddle the next packet) and
                            // wait for more bytes.
                            let keep = src.len().min(3);
                            let drop = src.len() - keep;
                            self.amount_skipped += drop;
                            src.advance(drop);
                            return Ok(None);
                        }
                    }
                }
            }
        };
        // Update context
        if let Bc {
            meta:
                BcMeta {
                    msg_id: 1,
                    response_code,
                    ..
                },
            body:
                BcBody::ModernMsg(ModernMsg {
                    payload:
                        Some(BcPayloads::BcXml(BcXml {
                            encryption: Some(Encryption { nonce, .. }),
                            ..
                        })),
                    ..
                }),
        } = &bc
        {
            if response_code >> 8 == 0xdd {
                // Login reply has the encryption info
                // Set that the encryption type now
                let encryption_protocol_byte = (response_code & 0xff) as usize;
                match encryption_protocol_byte {
                    0x00 => self.context.set_encrypted(EncryptionProtocol::Unencrypted),
                    0x01 => self.context.set_encrypted(EncryptionProtocol::BCEncrypt),
                    0x02 => self.context.set_encrypted(EncryptionProtocol::aes(
                        self.context.credentials.make_aeskey(nonce),
                    )),
                    0x12 => self.context.set_encrypted(EncryptionProtocol::full_aes(
                        self.context.credentials.make_aeskey(nonce),
                    )),
                    _ => {
                        return Err(Error::UnknownEncryption(encryption_protocol_byte));
                    }
                }
            }
        }

        if let BcBody::ModernMsg(ModernMsg {
            extension:
                Some(Extension {
                    binary_data: Some(on_off),
                    ..
                }),
            ..
        }) = bc.body
        {
            if on_off == 0 {
                self.context.binary_off(bc.meta.msg_num);
            } else {
                self.context.binary_on(bc.meta.msg_num);
            }
        }

        Ok(Some(bc))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bcencrypt_codex() -> BcCodex {
        BcCodex {
            context: BcContext::new_with_encryption(EncryptionProtocol::BCEncrypt),
            amount_skipped: 0,
        }
    }

    #[test]
    fn next_magic_skips_offset_zero() {
        // A valid magic sits at offset 0 (where a failed deserialize left us)
        // and the real next message starts at offset 8. We must skip past 0.
        let magic = MAGIC_HEADER.to_le_bytes();
        let mut buf = Vec::new();
        buf.extend_from_slice(&magic);
        buf.extend_from_slice(&[1, 2, 3, 4]);
        buf.extend_from_slice(&magic);
        buf.extend_from_slice(&[0, 0]);
        assert_eq!(next_bc_magic_offset(&buf), Some(8));
    }

    #[test]
    fn next_magic_finds_reversed_magic() {
        let magic = MAGIC_HEADER_REV.to_le_bytes();
        let mut buf = vec![0xAA, 0xBB, 0xCC, 0xDD, 0xEE];
        buf.extend_from_slice(&magic);
        buf.extend_from_slice(&[0, 0, 0]);
        assert_eq!(next_bc_magic_offset(&buf), Some(5));
    }

    #[test]
    fn next_magic_none_when_absent_or_short() {
        assert_eq!(next_bc_magic_offset(&[0u8; 32]), None);
        assert_eq!(next_bc_magic_offset(&[1, 2, 3]), None); // < 5 bytes
        assert_eq!(next_bc_magic_offset(&[]), None);
    }

    #[test]
    fn decode_resyncs_past_leading_garbage() {
        // A proven, fully-deserializable modern message under BCEncrypt.
        let sample = include_bytes!("samples/modern_video_start1.bin");

        // Sanity: decodes cleanly on its own.
        let mut clean = BytesMut::from(&sample[..]);
        assert!(bcencrypt_codex().decode(&mut clean).unwrap().is_some());

        // Simulate a hole from a skipped UDP packet: non-magic garbage in front
        // of an otherwise-intact message. The codec must skip the garbage and
        // still decode the message instead of erroring (which would drop the
        // whole connection).
        let mut buf = BytesMut::new();
        buf.extend_from_slice(&[0xAAu8; 37]);
        buf.extend_from_slice(&sample[..]);

        let mut codex = bcencrypt_codex();
        let msg = codex
            .decode(&mut buf)
            .expect("resync must not surface a fatal error");
        assert!(msg.is_some(), "should resync and decode the message");
        assert_eq!(codex.amount_skipped, 0, "skip counter resets after success");
    }
}
