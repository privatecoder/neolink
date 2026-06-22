use super::BcConnection;
use crate::bcmedia::codex::BcMediaCodex;
use crate::{bc::model::*, bcmedia::model::*, Error, Result};
use futures::stream::{Stream, TryStreamExt};
use std::io::{Error as IoError, Result as IoResult};
use tokio::sync::mpsc::Receiver;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};
use tokio_util::codec::FramedRead;
use tokio_util::compat::FuturesAsyncReadCompatExt;

pub struct BcSubscription<'a> {
    rx: ReceiverStream<Result<Bc>>,
    msg_num: Option<u32>,
    conn: &'a BcConnection,
}

impl<'a> BcSubscription<'a> {
    pub fn new(
        rx: Receiver<Result<Bc>>,
        msg_num: Option<u32>,
        conn: &'a BcConnection,
    ) -> BcSubscription<'a> {
        BcSubscription {
            rx: ReceiverStream::new(rx),
            msg_num,
            conn,
        }
    }

    pub async fn send(&self, bc: Bc) -> Result<()> {
        if let Some(msg_num) = self.msg_num {
            assert!(bc.meta.msg_num as u32 == msg_num);
        } else {
            log::debug!("Sending message before msg_num has been acquired");
        }
        self.conn.send(bc).await?;
        Ok(())
    }

    pub async fn recv(&mut self) -> Result<Bc> {
        let bc = self.rx.next().await.ok_or(Error::DroppedSubscriber)?;
        if let Ok(bc) = &bc {
            if let Some(msg_num) = self.msg_num {
                assert!(bc.meta.msg_num as u32 == msg_num);
            } else {
                // Leaning number now
                self.msg_num = Some(bc.meta.msg_num as u32);
            }
        }
        bc
    }

    pub fn payload_stream(&'_ mut self) -> impl Stream<Item = IoResult<Vec<u8>>> + '_ {
        (&mut self.rx).filter_map(|x| match x {
            Ok(Bc {
                meta: BcMeta { .. },
                body:
                    BcBody::ModernMsg(ModernMsg {
                        payload: Some(BcPayloads::Binary(data)),
                        ..
                    }),
            }) => Some(Ok(data)),
            Ok(_) => None,
            Err(e) => Some(Err(IoError::other(e))),
        })
    }

    pub fn bcmedia_stream(&'_ mut self, strict: bool) -> impl Stream<Item = Result<BcMedia>> + '_ {
        let async_read = self
            .payload_stream()
            .map(|frame| frame)
            .into_async_read()
            .compat();
        FramedRead::new(async_read, BcMediaCodex::new(strict)).map(|frame| frame)
    }
}
