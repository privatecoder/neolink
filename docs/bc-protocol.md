# BC control protocol (Baichuan)

The framing of the Baichuan control channel: every command and reply (login, PTZ,
battery, talk setup, stream requests, …) is a `Bc` packet. Implemented in
`crates/core/src/bc/` (`model.rs`, `de.rs`, `ser.rs`, `xml.rs`, `codex.rs`).

The control channel is carried over TCP, or tunnelled inside `UdpData` packets on the
P2P/relay paths (see [discovery-handshake.md](discovery-handshake.md)). The media
substream is a separate format — see [media-streams.md](media-streams.md).

## Packet layout

A `Bc` packet is a fixed header followed by a body. All multi-byte fields are
**little-endian**.

| Offset | Field | Size | Notes |
|---|---|---|---|
| 0 | magic | 4 | `0x0abcdef0` (or reversed `0x0fedcba0`, accepted on decode, seen on some JPEG/snap replies) |
| 4 | `msg_id` | 4 | message type (see table below) |
| 8 | `body_len` | 4 | length of the body |
| 12 | `channel_id` | 1 | camera channel; also the per-message encryption offset |
| 13 | `stream_type` | 1 | `0` = Clear, `1` = Fluent (used with video) |
| 14 | `msg_num` | 2 | request/reply correlation; shared across split packets |
| 16 | `response_code` | 2 | `0` on a request; `200` OK / `400` bad request on a reply |
| 18 | `class` | 2 | selects framing variant (below) |
| 20 | `payload_offset` | 0 or 4 | present only when `class` is `0x6414` or `0x0000` |

Header is 20 bytes, or 24 with `payload_offset`. Only the canonical magic is written
on encode.

### Class values

| `class` | Variant |
|---|---|
| `0x6514` | **legacy**, 20-byte header (initial login message) |
| `0x6614` | modern, 20-byte header (reply to an encrypted login) |
| `0x6414` | modern, 24-byte header with payload offset (re-sent login) |
| `0x0000` | modern, 24-byte header with payload offset (most modern messages) |

## Modern vs legacy

`is_modern()` is `class != 0x6514` — only `0x6514` is legacy. Legacy is used solely
for login: a `LegacyMsg::LoginMsg { username, password }` whose body is two 32-byte
hex fields padded to a fixed 1836-byte body.

A modern message (`ModernMsg`) has two optional parts split by `payload_offset`:

- **extension** — an XML block describing the payload (e.g. `binaryData`,
  `encryptLen`, channel/token). Present when `payload_offset > 0`.
- **payload** — the bytes from `payload_offset` to `body_len`; either an XML body
  (`BcXml`) or raw `Binary`. Absent when `payload_offset == body_len`.

Both `None` ⇒ a header-only acknowledgement (inspect `response_code`).

## XML bodies

`BcXml` (`xml.rs`) is one large struct of optional sub-elements — typically exactly
one is populated per message (`Encryption`, `LoginUser`, `DeviceInfo`, `Preview`,
`BatteryList`, `Snap`, `TalkAbility`, …). It serialises under the root tag `body`
with an XML declaration; `Extension` serialises under `Extension`. Parsing/serialising
go through `quick_xml`.

## Binary mode

Streams (camera → client) and talk/firmware (client → camera) carry raw bytes as
`BcPayloads::Binary` rather than XML. Binary mode is declared by the extension's
`binaryData = 1`; subsequent split packets with the same `msg_num` carry no extension,
so the decoder records "this `msg_num` is binary" in a per-context set
(`in_bin_mode`) and decodes the continuation packets as binary. Under `FullAes`, the
extension's `encryptLen` bounds the decrypted binary length (with `checkPos`/
`checkValue` to validate decryption).

## Framing (Encoder / Decoder)

`BcCodex` is a `tokio_util` `Encoder<Bc>` / `Decoder`:

- **Decode** uses streaming nom parsers. If a packet is incomplete it returns
  `Ok(None)` (via `Error::NomIncomplete`) so the framer waits for more bytes; leftover
  bytes at EOF are logged, not errored.
- After decoding a login reply it updates the session encryption (below).
- **Encode** serialises the `Bc` and, for `msg_id == 1` (login) while the session is
  AES/FullAes, downgrades that one message to BCEncrypt (the AES nonce is not yet
  established during login).

## Encryption

The control channel is encrypted; the level is negotiated at login and capped
per-camera by `max_encryption` (`none` | `bcencrypt` | `aes`, default `aes`).
Implemented in `crates/core/src/bc/crypto.rs`.

| Mode | Algorithm |
|---|---|
| Unencrypted | none |
| BCEncrypt | byte-wise XOR against a fixed 8-byte key, offset by the packet's `channel_id` |
| AES | AES-128-CFB (fixed IV); control channel only |
| FullAes | AES-128-CFB (fixed IV); control channel **and** media stream |

- The AES key is derived (MD5-based) from the camera password and a per-session nonce
  exchanged during login.
- **Negotiation:** the login reply has `msg_id == 1` and `response_code >> 8 == 0xdd`;
  its low byte selects the mode — `0x00` Unencrypted, `0x01` BCEncrypt, `0x02` AES,
  `0x12` FullAes. An unknown byte is an error.
- During the login exchange the maximum is BCEncrypt (enforced on both decode and
  encode) because the nonce has not yet been exchanged; AES/FullAes take effect
  afterwards.

## Message IDs (selected)

`MSG_ID_*` constants (`model.rs`):

| id | name | id | name |
|---|---|---|---|
| 1 | LOGIN | 109 | SNAP |
| 2 | LOGOUT | 114 | UID |
| 3 | VIDEO | 146 | STREAM_INFO_LIST |
| 4 | VIDEO_STOP | 151 | ABILITY_INFO |
| 10 | TALKABILITY | 201 | TALKCONFIG |
| 11 | TALKRESET | 202 | TALK |
| 18/19 | PTZ_CONTROL / _PRESET | 208/209 | GET/SET_LED_STATUS |
| 23 | REBOOT | 212/213 | GET/START_PIR_ALARM |
| 31/33 | MOTION_REQUEST / MOTION | 234 | UDP_KEEP_ALIVE |
| 36/37 | SET/GET_SERVICE_PORTS | 252/253 | BATTERY_INFO_LIST / _INFO |
| 58 | GET_ABILITY_SUPPORT | 263 | PLAY_AUDIO |
| 80 | VERSION | 288–291 | FLOODLIGHT_* |
| 93 | PING | 294/295 | GET/SET_ZOOM_FOCUS |
| 104/105 | GET/SET_GENERAL | 438 | FLOODLIGHT_TASKS_READ |

(See `model.rs` for the full list.)
