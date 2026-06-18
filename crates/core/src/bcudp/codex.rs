//! Handles sending and recieving messages as complete packets
//!
//! BcUdpCodex is used with a `[tokio_util::codec::Framed]` to form complete packets
//!
use crate::bcudp::model::*;
use crate::{Error, Result};
use bytes::BytesMut;
use tokio_util::codec::{Decoder, Encoder};

use super::xml::UdpXml;

pub(crate) struct BcUdpCodex {}

impl BcUdpCodex {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl Encoder<BcUdp> for BcUdpCodex {
    type Error = Error;

    fn encode(&mut self, item: BcUdp, dst: &mut BytesMut) -> Result<()> {
        log::trace!("Encoding: {item:?}");
        let buf: Vec<u8> = Default::default();
        let buf = item.serialize(buf)?;
        dst.extend_from_slice(buf.as_slice());
        log::trace!("  Encoding: Done: {}", buf.len());
        Ok(())
    }
}

impl Decoder for BcUdpCodex {
    type Item = BcUdp;
    type Error = Error;

    fn decode(&mut self, src: &mut BytesMut) -> Result<Option<Self::Item>> {
        log::trace!("Decoding:");
        if src.is_empty() {
            return Ok(None);
        }
        match BcUdp::deserialize(src) {
            Ok(BcUdp::Discovery(UdpDiscovery {
                payload: UdpXml::R2cDisc(_),
                ..
            })) => {
                log::trace!("   Decoding: Relay terminate");
                Err(Error::RelayTerminate)
            }
            Ok(BcUdp::Discovery(UdpDiscovery {
                payload: UdpXml::D2cDisc(_),
                ..
            })) => {
                log::trace!("   Decoding:Camera terminate");
                Err(Error::CameraTerminate)
            }
            Ok(bc) => {
                log::trace!("   Decoding: Ok");
                Ok(Some(bc))
            }
            Err(Error::NomIncomplete(_)) => {
                log::trace!("   Decoding: Incomplete: {:0X?}", src);
                Ok(None)
            }
            Err(e) => {
                log::trace!("   Decoding: Err");
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_matches::assert_matches;

    #[test]
    fn foreign_datagram_decodes_to_nom_error() {
        // A real captured stray datagram beginning with "NEXU". Its first 4 bytes
        // match none of the three BCUDP magics (0x2a87cf3a / 0x2a87cf20 /
        // 0x2a87cf10), so the magic `verify` fails and decoding rejects it as a
        // NomError (not NomIncomplete, since the failure is at the 4-byte magic).
        let bytes = [
            0x4E, 0x45, 0x58, 0x55, 0x01, 0x01, 0x0C, 0xC5, 0x74, 0xEF, 0x60, 0xF3, 0xE1, 0x14,
            0x00, 0x00,
        ];
        let mut buf = BytesMut::from(&bytes[..]);

        let result = BcUdpCodex::new().decode(&mut buf);

        assert_matches!(result, Err(Error::NomError(_)));
    }
}
