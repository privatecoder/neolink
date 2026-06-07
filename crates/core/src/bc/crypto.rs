use aes::{cipher::KeyIvInit, Aes128};
use cfb_mode::{Decryptor, Encryptor};

type Aes128CfbEnc = Encryptor<Aes128>;
type Aes128CfbDec = Decryptor<Aes128>;

const XML_KEY: [u8; 8] = [0x1F, 0x2D, 0x3C, 0x4B, 0x5A, 0x69, 0x78, 0xFF];
const IV: &[u8] = b"0123456789abcdef";

/// These are the encyption modes supported by the camera
///
/// The mode is negotiated during login
#[derive(Debug, Clone)]
pub enum EncryptionProtocol {
    /// Older camera use no encryption
    Unencrypted,
    /// Camera/Firmwares before 2021 use BCEncrypt which is a simple XOr
    BCEncrypt,
    /// Latest cameras/firmwares use Aes with the key derived from
    /// the camera's password and the negotiated NONCE
    Aes {
        /// The 16-byte AES-128 key; the CFB cipher is built per call
        key: [u8; 16],
    },
    /// Same as Aes but the media stream is also encrypted and not just
    /// the control commands
    FullAes {
        /// The 16-byte AES-128 key; the CFB cipher is built per call
        key: [u8; 16],
    },
}

impl EncryptionProtocol {
    /// Helper to make unencrypted
    pub fn unencrypted() -> Self {
        EncryptionProtocol::Unencrypted
    }
    /// Helper to make bcencrypted
    pub fn bcencrypt() -> Self {
        EncryptionProtocol::BCEncrypt
    }
    /// Helper to make aes
    pub fn aes(key: [u8; 16]) -> Self {
        EncryptionProtocol::Aes { key }
    }
    /// Helper to make full aes
    pub fn full_aes(key: [u8; 16]) -> Self {
        EncryptionProtocol::FullAes { key }
    }

    /// Decrypt the data, offset comes from the header of the packet
    pub fn decrypt(&self, offset: u32, buf: &[u8]) -> Vec<u8> {
        match self {
            EncryptionProtocol::Unencrypted => buf.to_vec(),
            EncryptionProtocol::BCEncrypt => {
                let key_iter = XML_KEY.iter().cycle().skip(offset as usize % 8);
                key_iter
                    .zip(buf)
                    .map(|(key, i)| *i ^ key ^ (offset as u8))
                    .collect()
            }
            EncryptionProtocol::Aes { key } | EncryptionProtocol::FullAes { key } => {
                // AES decryption

                let mut decrypted = buf.to_vec();
                let dec = Aes128CfbDec::new_from_slices(key, IV)
                    .expect("AES-128 key and IV are fixed valid lengths");
                dec.decrypt(&mut decrypted);
                decrypted
            }
        }
    }

    /// Encrypt the data, offset comes from the header of the packet
    pub fn encrypt(&self, offset: u32, buf: &[u8]) -> Vec<u8> {
        match self {
            EncryptionProtocol::Unencrypted => {
                // Encrypt is the same as decrypt
                self.decrypt(offset, buf)
            }
            EncryptionProtocol::BCEncrypt => {
                // Encrypt is the same as decrypt
                self.decrypt(offset, buf)
            }
            EncryptionProtocol::Aes { key } | EncryptionProtocol::FullAes { key } => {
                // AES encryption
                let mut encrypted = buf.to_vec();
                let enc = Aes128CfbEnc::new_from_slices(key, IV)
                    .expect("AES-128 key and IV are fixed valid lengths");
                enc.encrypt(&mut encrypted);
                encrypted
            }
        }
    }
}

#[test]
fn test_xml_crypto() {
    let sample = include_bytes!("samples/xml_crypto_sample1.bin");
    let should_be = include_bytes!("samples/xml_crypto_sample1_plaintext.bin");

    let decrypted = EncryptionProtocol::BCEncrypt.decrypt(0, &sample[..]);
    assert_eq!(decrypted, &should_be[..]);
}

#[test]
fn test_xml_crypto_roundtrip() {
    let zeros: [u8; 256] = [0; 256];

    let decrypted = EncryptionProtocol::BCEncrypt.encrypt(0, &zeros[..]);
    let encrypted = EncryptionProtocol::BCEncrypt.decrypt(0, &decrypted[..]);
    assert_eq!(encrypted, &zeros[..]);
}
