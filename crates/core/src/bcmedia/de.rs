use super::model::*;
use crate::Error;
use bytes::{Buf, BytesMut};
use nom::{bytes::streaming::take, combinator::*, error::context, number::streaming::*, Parser};

type IResult<I, O, E = nom_language::error::VerboseError<I>> = Result<(I, O), nom::Err<E>>;

// PAD_SIZE: Media packets use 8 byte padding
const PAD_SIZE: u32 = 8;

/// Upper bound on a single media frame's `payload_size` / `additional_header_size`,
/// enforced before the `take` that would buffer them. Both are wire-supplied `u32`s,
/// so a bogus media header could otherwise drive the framing layer to buffer toward a
/// multi-gigabyte declared size and exhaust memory (DoS), on both the TCP and UDP
/// transports. 16 MiB is far above any real frame — a 4K key-frame is a few MiB at
/// most, and the `additional_header` is only a handful of bytes — while still
/// rejecting clearly malicious lengths.
const MAX_MEDIA_PAYLOAD: u32 = 16 * 1024 * 1024;

impl BcMedia {
    pub(crate) fn deserialize(buf: &mut BytesMut) -> Result<BcMedia, Error> {
        let (result, len) = match consumed(bcmedia).parse(buf) {
            Ok((_, (parsed_buff, result))) => Ok((result, parsed_buff.len())),
            Err(e) => Err(e),
        }?;
        buf.advance(len);
        Ok(result)
    }
}

fn bcmedia(buf: &[u8]) -> IResult<&[u8], BcMedia> {
    let (buf, magic) = context(
        "Failed to match any known bcmedia",
        verify(le_u32, |x| {
            matches!(
                *x,
                MAGIC_HEADER_BCMEDIA_INFO_V1
                    | MAGIC_HEADER_BCMEDIA_INFO_V2
                    | MAGIC_HEADER_BCMEDIA_IFRAME..=MAGIC_HEADER_BCMEDIA_IFRAME_LAST
                    | MAGIC_HEADER_BCMEDIA_PFRAME..=MAGIC_HEADER_BCMEDIA_PFRAME_LAST
                    | MAGIC_HEADER_BCMEDIA_AAC
                    | MAGIC_HEADER_BCMEDIA_ADPCM
            )
        }),
    )
    .parse(buf)?;

    match magic {
        MAGIC_HEADER_BCMEDIA_INFO_V1 => {
            let (buf, payload) = bcmedia_info_v1(buf)?;
            Ok((buf, BcMedia::InfoV1(payload)))
        }
        MAGIC_HEADER_BCMEDIA_INFO_V2 => {
            let (buf, payload) = bcmedia_info_v2(buf)?;
            Ok((buf, BcMedia::InfoV2(payload)))
        }
        MAGIC_HEADER_BCMEDIA_IFRAME..=MAGIC_HEADER_BCMEDIA_IFRAME_LAST => {
            let (buf, payload) = bcmedia_iframe(buf)?;
            Ok((buf, BcMedia::Iframe(payload)))
        }
        MAGIC_HEADER_BCMEDIA_PFRAME..=MAGIC_HEADER_BCMEDIA_PFRAME_LAST => {
            let (buf, payload) = bcmedia_pframe(buf)?;
            Ok((buf, BcMedia::Pframe(payload)))
        }
        MAGIC_HEADER_BCMEDIA_AAC => {
            let (buf, payload) = bcmedia_aac(buf)?;
            Ok((buf, BcMedia::Aac(payload)))
        }
        MAGIC_HEADER_BCMEDIA_ADPCM => {
            let (buf, payload) = bcmedia_adpcm(buf)?;
            Ok((buf, BcMedia::Adpcm(payload)))
        }
        _ => unreachable!(),
    }
}

fn bcmedia_info_v1(buf: &[u8]) -> IResult<&[u8], BcMediaInfoV1> {
    let (buf, _header_size) = context(
        "Header size mismatch in BCMedia InfoV1",
        verify(le_u32, |x| *x == 32),
    )
    .parse(buf)?;
    let (buf, video_width) = le_u32(buf)?;
    let (buf, video_height) = le_u32(buf)?;
    let (buf, _unknown) = le_u8(buf)?;
    let (buf, fps) = le_u8(buf)?;
    let (buf, start_year) = le_u8(buf)?;
    let (buf, start_month) = le_u8(buf)?;
    let (buf, start_day) = le_u8(buf)?;
    let (buf, start_hour) = le_u8(buf)?;
    let (buf, start_min) = le_u8(buf)?;
    let (buf, start_seconds) = le_u8(buf)?;
    let (buf, end_year) = le_u8(buf)?;
    let (buf, end_month) = le_u8(buf)?;
    let (buf, end_day) = le_u8(buf)?;
    let (buf, end_hour) = le_u8(buf)?;
    let (buf, end_min) = le_u8(buf)?;
    let (buf, end_seconds) = le_u8(buf)?;
    let (buf, _unknown_b) = le_u16(buf)?;

    Ok((
        buf,
        BcMediaInfoV1 {
            // header_size,
            video_width,
            video_height,
            fps,
            start_year,
            start_month,
            start_day,
            start_hour,
            start_min,
            start_seconds,
            end_year,
            end_month,
            end_day,
            end_hour,
            end_min,
            end_seconds,
        },
    ))
}

fn bcmedia_info_v2(buf: &[u8]) -> IResult<&[u8], BcMediaInfoV2> {
    let (buf, _header_size) = context(
        "Failed to match headersize in BCMedia Info V2",
        verify(le_u32, |x| *x == 32),
    )
    .parse(buf)?;
    let (buf, video_width) = le_u32(buf)?;
    let (buf, video_height) = le_u32(buf)?;
    let (buf, _unknown) = le_u8(buf)?;
    let (buf, fps) = le_u8(buf)?;
    let (buf, start_year) = le_u8(buf)?;
    let (buf, start_month) = le_u8(buf)?;
    let (buf, start_day) = le_u8(buf)?;
    let (buf, start_hour) = le_u8(buf)?;
    let (buf, start_min) = le_u8(buf)?;
    let (buf, start_seconds) = le_u8(buf)?;
    let (buf, end_year) = le_u8(buf)?;
    let (buf, end_month) = le_u8(buf)?;
    let (buf, end_day) = le_u8(buf)?;
    let (buf, end_hour) = le_u8(buf)?;
    let (buf, end_min) = le_u8(buf)?;
    let (buf, end_seconds) = le_u8(buf)?;
    let (buf, _unknown_b) = le_u16(buf)?;

    Ok((
        buf,
        BcMediaInfoV2 {
            // header_size,
            video_width,
            video_height,
            fps,
            start_year,
            start_month,
            start_day,
            start_hour,
            start_min,
            start_seconds,
            end_year,
            end_month,
            end_day,
            end_hour,
            end_min,
            end_seconds,
        },
    ))
}

fn take4(buf: &[u8]) -> IResult<&[u8], &str> {
    map_res(nom::bytes::streaming::take(4usize), |r| {
        std::str::from_utf8(r)
    })
    .parse(buf)
}

fn bcmedia_iframe(buf: &[u8]) -> IResult<&[u8], BcMediaIframe> {
    let (buf, video_type_str) = context(
        "Video Type is unrecognised in IFrame",
        verify(take4, |x| matches!(x, "H264" | "H265")),
    )
    .parse(buf)?;
    // Reject oversized wire lengths before the `take`s below buffer toward them (DoS).
    let (buf, payload_size) = context(
        "IFrame payload_size exceeds maximum",
        verify(le_u32, |x: &u32| *x <= MAX_MEDIA_PAYLOAD),
    )
    .parse(buf)?;
    let (buf, additional_header_size) = context(
        "IFrame additional_header_size exceeds maximum",
        verify(le_u32, |x: &u32| *x <= MAX_MEDIA_PAYLOAD),
    )
    .parse(buf)?;
    let (buf, microseconds) = le_u32(buf)?;
    let (buf, _unknown_b) = le_u32(buf)?;
    let (buf, time) = if additional_header_size >= 4 {
        let (buf, time_value) = le_u32(buf)?;
        (buf, Some(time_value))
    } else {
        (buf, None)
    };
    let (buf, _unknown_remained) = if additional_header_size > 4 {
        let remainder = additional_header_size - 4;
        let (buf, unknown_remained) = take(remainder)(buf)?;
        (buf, Some(unknown_remained))
    } else {
        (buf, None)
    };

    let (buf, data_slice) = take(payload_size)(buf)?;
    let pad_size = match payload_size % PAD_SIZE {
        0 => 0,
        n => PAD_SIZE - n,
    };
    let (buf, _padding) = take(pad_size)(buf)?;
    assert_eq!(payload_size as usize, data_slice.len());

    let video_type = match video_type_str {
        "H264" => VideoType::H264,
        "H265" => VideoType::H265,
        _ => unreachable!(),
    };

    Ok((
        buf,
        BcMediaIframe {
            video_type,
            // payload_size,
            microseconds,
            time,
            data: data_slice.to_vec(),
        },
    ))
}

fn bcmedia_pframe(buf: &[u8]) -> IResult<&[u8], BcMediaPframe> {
    let (buf, video_type_str) = context(
        "Video Type is unrecognised in PFrame",
        verify(take4, |x| matches!(x, "H264" | "H265")),
    )
    .parse(buf)?;
    // Reject oversized wire lengths before the `take`s below buffer toward them (DoS).
    let (buf, payload_size) = context(
        "PFrame payload_size exceeds maximum",
        verify(le_u32, |x: &u32| *x <= MAX_MEDIA_PAYLOAD),
    )
    .parse(buf)?;
    let (buf, additional_header_size) = context(
        "PFrame additional_header_size exceeds maximum",
        verify(le_u32, |x: &u32| *x <= MAX_MEDIA_PAYLOAD),
    )
    .parse(buf)?;
    let (buf, microseconds) = le_u32(buf)?;
    let (buf, _unknown_b) = le_u32(buf)?;
    let (buf, _additional_header) = take(additional_header_size)(buf)?;
    let (buf, data_slice) = take(payload_size)(buf)?;
    let pad_size = match payload_size % PAD_SIZE {
        0 => 0,
        n => PAD_SIZE - n,
    };
    let (buf, _padding) = take(pad_size)(buf)?;
    assert_eq!(payload_size as usize, data_slice.len());

    let video_type = match video_type_str {
        "H264" => VideoType::H264,
        "H265" => VideoType::H265,
        _ => unreachable!(),
    };

    Ok((
        buf,
        BcMediaPframe {
            video_type,
            // payload_size,
            microseconds,
            data: data_slice.to_vec(),
        },
    ))
}

fn bcmedia_aac(buf: &[u8]) -> IResult<&[u8], BcMediaAac> {
    let (buf, payload_size) = le_u16(buf)?;
    let (buf, _payload_size_b) = le_u16(buf)?;
    let (buf, data_slice) = take(payload_size)(buf)?;
    let pad_size = match payload_size as u32 % PAD_SIZE {
        0 => 0,
        n => PAD_SIZE - n,
    };
    let (buf, _padding) = take(pad_size)(buf)?;

    Ok((
        buf,
        BcMediaAac {
            // payload_size,
            data: data_slice.to_vec(),
        },
    ))
}

fn bcmedia_adpcm(buf: &[u8]) -> IResult<&[u8], BcMediaAdpcm> {
    const SUB_HEADER_SIZE: u16 = 4;

    // `block_size` below is `payload_size - SUB_HEADER_SIZE`; reject a wire-supplied
    // `payload_size` smaller than the sub-header so the subtraction cannot underflow
    // (panic in debug / huge wrap + over-long `take` in release).
    let (buf, payload_size) = context(
        "ADPCM payload_size smaller than sub-header",
        verify(le_u16, |x: &u16| *x >= SUB_HEADER_SIZE),
    )
    .parse(buf)?;
    let (buf, _payload_size_b) = le_u16(buf)?;
    let (buf, _magic) = context(
        "ADPCM data magic value is invalid",
        verify(le_u16, |x| *x == MAGIC_HEADER_BCMEDIA_ADPCM_DATA),
    )
    .parse(buf)?;
    // On some camera this value is just 2
    // On other cameras is half the block size without the header
    let (buf, _half_block_size) = le_u16(buf)?;
    let block_size = payload_size - SUB_HEADER_SIZE;
    let (buf, data_slice) = take(block_size)(buf)?;
    let pad_size = match payload_size as u32 % PAD_SIZE {
        0 => 0,
        n => PAD_SIZE - n,
    };
    let (buf, _padding) = take(pad_size)(buf)?;

    Ok((
        buf,
        BcMediaAdpcm {
            // payload_size,
            // block_size,
            data: data_slice.to_vec(),
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::Error;
    use crate::bcmedia::model::*;
    use bytes::BytesMut;
    use env_logger::Env;
    use log::*;
    use std::io::ErrorKind;

    fn init() {
        let _ = env_logger::Builder::from_env(Env::default().default_filter_or("info"))
            .is_test(true)
            .try_init();
    }

    #[test]
    // This method will test the decoding on swann cameras output
    //
    // Crucially this contains adpcm
    fn test_swan_deser() {
        init();

        let sample = [
            include_bytes!("samples/video_stream_swan_00.raw").as_ref(),
            include_bytes!("samples/video_stream_swan_01.raw").as_ref(),
            include_bytes!("samples/video_stream_swan_02.raw").as_ref(),
            include_bytes!("samples/video_stream_swan_03.raw").as_ref(),
            include_bytes!("samples/video_stream_swan_04.raw").as_ref(),
            include_bytes!("samples/video_stream_swan_05.raw").as_ref(),
            include_bytes!("samples/video_stream_swan_06.raw").as_ref(),
            include_bytes!("samples/video_stream_swan_07.raw").as_ref(),
            include_bytes!("samples/video_stream_swan_08.raw").as_ref(),
            include_bytes!("samples/video_stream_swan_09.raw").as_ref(),
        ]
        .concat();

        let mut buf = BytesMut::from(&sample[..]);

        // Should derealise all of this
        loop {
            let e = BcMedia::deserialize(&mut buf);
            match e {
                Err(Error::Io(e)) if e.kind() == ErrorKind::UnexpectedEof => {
                    // Reach end of files
                    break;
                }
                Err(Error::NomIncomplete(_)) if buf.is_empty() => {
                    // EOF still (but parser looking for next magic)
                    break;
                }
                Err(e) => {
                    error!("{:?}", e);
                    panic!();
                }
                Ok(_) => {}
            }
        }
    }

    #[test]
    // This method will test the decoding of argus2 cameras output
    //
    // This packet has an extended iframe
    fn test_argus2_iframe_extended() {
        init();

        let sample = [
            include_bytes!("samples/argus2_iframe_0.raw").as_ref(),
            include_bytes!("samples/argus2_iframe_1.raw").as_ref(),
            include_bytes!("samples/argus2_iframe_2.raw").as_ref(),
            include_bytes!("samples/argus2_iframe_3.raw").as_ref(),
            include_bytes!("samples/argus2_iframe_4.raw").as_ref(),
        ]
        .concat();

        let mut buf = BytesMut::from(&sample[..]);
        // Should derealise all of this
        loop {
            let e = BcMedia::deserialize(&mut buf);
            match e {
                Err(Error::Io(e)) if e.kind() == ErrorKind::UnexpectedEof => {
                    // Reach end of files
                    break;
                }
                Err(Error::NomIncomplete(_)) if buf.is_empty() => {
                    // EOF still (but parser looking for next magic)
                    break;
                }
                Err(e) => {
                    error!("{:?}", e);
                    panic!();
                }
                Ok(_) => {}
            }
        }
    }

    #[test]
    // This method will test the decoding of argus2 cameras output
    //
    // This packet has an extended pframe
    fn test_argus2_pframe_extended() {
        init();

        let sample = [
            include_bytes!("samples/argus2_pframe_0.raw").as_ref(),
            include_bytes!("samples/argus2_pframe_1.raw").as_ref(),
            include_bytes!("samples/argus2_pframe_2.raw").as_ref(),
            include_bytes!("samples/argus2_pframe_3.raw").as_ref(),
            include_bytes!("samples/argus2_pframe_4.raw").as_ref(),
            include_bytes!("samples/argus2_pframe_5.raw").as_ref(),
            include_bytes!("samples/argus2_pframe_6.raw").as_ref(),
            include_bytes!("samples/argus2_pframe_7.raw").as_ref(),
            include_bytes!("samples/argus2_pframe_8.raw").as_ref(),
            include_bytes!("samples/argus2_pframe_9.raw").as_ref(),
            include_bytes!("samples/argus2_pframe_10.raw").as_ref(),
            include_bytes!("samples/argus2_pframe_11.raw").as_ref(),
            include_bytes!("samples/argus2_pframe_12.raw").as_ref(),
            include_bytes!("samples/argus2_pframe_13.raw").as_ref(),
            include_bytes!("samples/argus2_pframe_14.raw").as_ref(),
            include_bytes!("samples/argus2_pframe_15.raw").as_ref(),
            include_bytes!("samples/argus2_pframe_16.raw").as_ref(),
            include_bytes!("samples/argus2_pframe_17.raw").as_ref(),
        ]
        .concat();

        let mut buf = BytesMut::from(&sample[..]);

        // Should derealise all of this
        loop {
            let e = BcMedia::deserialize(&mut buf);
            match e {
                Err(Error::Io(e)) if e.kind() == ErrorKind::UnexpectedEof => {
                    // Reach end of files
                    break;
                }
                Err(Error::NomIncomplete(_)) if buf.is_empty() => {
                    // EOF still (but parser looking for next magic)
                    break;
                }
                Err(e) => {
                    error!("{:?}", e);
                    panic!();
                }
                Ok(_) => {}
            }
        }
    }

    #[test]
    // Tests the decoding of an info v1
    fn test_info_v1() {
        init();

        let sample = include_bytes!("samples/info_v1.raw");

        let mut buf = BytesMut::from(&sample[..]);

        let e = BcMedia::deserialize(&mut buf);
        assert!(matches!(
            e,
            Ok(BcMedia::InfoV1(BcMediaInfoV1 {
                video_width: 2560,
                video_height: 1440,
                fps: 30,
                start_year: 121,
                start_month: 8,
                start_day: 4,
                start_hour: 23,
                start_min: 23,
                start_seconds: 52,
                end_year: 121,
                end_month: 8,
                end_day: 4,
                end_hour: 23,
                end_min: 23,
                end_seconds: 52,
            }))
        ));
    }

    #[test]
    fn test_iframe() {
        init();

        let sample = [
            include_bytes!("samples/iframe_0.raw").as_ref(),
            include_bytes!("samples/iframe_1.raw").as_ref(),
            include_bytes!("samples/iframe_2.raw").as_ref(),
            include_bytes!("samples/iframe_3.raw").as_ref(),
            include_bytes!("samples/iframe_4.raw").as_ref(),
        ]
        .concat();

        let mut buf = BytesMut::from(&sample[..]);

        let e = BcMedia::deserialize(&mut buf);
        if let Ok(BcMedia::Iframe(BcMediaIframe {
            video_type: VideoType::H264,
            microseconds: 3557705112,
            time: Some(1628085232),
            data: d,
        })) = e
        {
            assert_eq!(d.len(), 192881);
        } else {
            panic!();
        }
    }

    #[test]
    fn test_pframe() {
        init();

        let sample = [
            include_bytes!("samples/pframe_0.raw").as_ref(),
            include_bytes!("samples/pframe_1.raw").as_ref(),
        ]
        .concat();

        let mut buf = BytesMut::from(&sample[..]);

        let e = BcMedia::deserialize(&mut buf);
        if let Ok(BcMedia::Pframe(BcMediaPframe {
            video_type: VideoType::H264,
            microseconds: 3557767112,
            data: d,
        })) = e
        {
            assert_eq!(d.len(), 45108);
        } else {
            panic!();
        }
    }

    #[test]
    fn test_adpcm() {
        init();

        let sample = include_bytes!("samples/adpcm_0.raw");
        let mut buf = BytesMut::from(&sample[..]);

        let e = BcMedia::deserialize(&mut buf);
        if let Ok(BcMedia::Adpcm(BcMediaAdpcm { data: d })) = e {
            assert_eq!(d.len(), 244);
        } else {
            panic!();
        }
    }

    #[test]
    // A wire-supplied ADPCM payload_size below the sub-header size must be a parse
    // error (not an underflow panic / over-long take). See `bcmedia_adpcm`.
    fn rejects_adpcm_payload_smaller_than_subheader() {
        init();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC_HEADER_BCMEDIA_ADPCM.to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes()); // payload_size < SUB_HEADER_SIZE (4)
        bytes.extend_from_slice(&0u16.to_le_bytes()); // payload_size_b
        match BcMedia::deserialize(&mut BytesMut::from(&bytes[..])) {
            Ok(_) => panic!("expected a parse error"),
            Err(Error::NomIncomplete(_)) => panic!("expected a hard error, got incomplete"),
            Err(_) => {}
        }
    }

    #[test]
    // An oversized wire payload_size must be rejected before the take buffers
    // toward it (DoS). See `MAX_MEDIA_PAYLOAD`.
    fn rejects_oversize_iframe_payload() {
        init();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC_HEADER_BCMEDIA_IFRAME.to_le_bytes());
        bytes.extend_from_slice(b"H264");
        bytes.extend_from_slice(&(super::MAX_MEDIA_PAYLOAD + 1).to_le_bytes()); // payload_size
        bytes.extend_from_slice(&0u32.to_le_bytes()); // additional_header_size
        match BcMedia::deserialize(&mut BytesMut::from(&bytes[..])) {
            Ok(_) => panic!("expected a parse error"),
            Err(Error::NomIncomplete(_)) => panic!("expected a hard error, got incomplete"),
            Err(_) => {}
        }
    }
}
