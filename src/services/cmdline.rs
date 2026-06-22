use anyhow::{anyhow, Result};
use clap::{Parser, ValueEnum};

fn onoff_parse(src: &str) -> Result<bool> {
    match src.trim().to_lowercase().as_str() {
        "true" | "on" | "yes" | "1" => Ok(true),
        "false" | "off" | "no" | "0" => Ok(false),
        _ => Err(anyhow!(
            "Could not understand {}, check your input, should be true/false, on/off or yes/no",
            src
        )),
    }
}

/// The services command will control the ports for http/https/rtmp/rtsp/onvif
#[derive(Parser, Debug)]
pub struct Opt {
    /// The name of the camera. Must be a name in the config
    pub camera: String,
    /// service to change
    pub service: Services,
    /// The action to perform
    #[command(subcommand)]
    pub cmd: PortAction,
}

#[derive(Parser, Debug, Clone, ValueEnum)]
pub enum Services {
    Baichuan,
    Http,
    Https,
    Rtmp,
    Rtsp,
    Onvif,
}

#[derive(Parser, Debug)]
pub enum PortAction {
    /// Get the current details
    Get,
    /// Turn the service ON
    On,
    /// Turn the service OFF
    Off,
    /// Set both port and on/off
    Set {
        /// The new port
        port: u16,
        /// On/off
        #[arg(value_parser = onoff_parse, action = clap::ArgAction::Set, name = "on|off")]
        enabled: bool,
    },
    /// Set the port
    Port {
        /// The new port
        port: u16,
    },
}
