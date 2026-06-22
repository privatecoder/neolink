use anyhow::{anyhow, Result};
use clap::Parser;

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

/// The pir command will control the PIR status of the camera
#[derive(Parser, Debug)]
pub struct Opt {
    /// The name of the camera. Must be a name in the config
    pub camera: String,
    /// Whether to turn the PIR ON or OFF
    #[arg(value_parser = onoff_parse, action = clap::ArgAction::Set, name = "on|off")]
    pub on: Option<bool>,
}
