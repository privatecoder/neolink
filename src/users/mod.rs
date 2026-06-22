///
/// # Neolink Users
///
/// This modules handles users
///
///
/// # Usage
///
/// ```bash
/// # To list users
/// neolink --config=config.toml users CameraName list
/// # To create a a new administrator
/// neolink --config=config.toml users CameraName add newuser hunter2 administrator
/// # To delete a user
/// neolink --config=config.toml users CameraName delete newuser
/// ```
///
/// There are two types of users:
/// - administrator — Can access the device and change device settings
/// - user — Can access the device but not change device settings
///
use anyhow::{anyhow, Context, Result};

mod cmdline;

use crate::common::NeoReactor;
pub(crate) use cmdline::*;
use neolink_core::bc::xml::UserList;

/// Entry point for the users subcommand
///
/// Opt is the command line options
pub(crate) async fn main(opt: Opt, reactor: NeoReactor) -> Result<()> {
    let camera = reactor.get(&opt.camera).await?;

    match opt.cmd {
        UserAction::List => {
            let user_list = camera
                .run_task(|cam| {
                    Box::pin(
                        async move { cam.get_users().await.context("Unable to list camera users") },
                    )
                })
                .await?;
            match user_list {
                UserList {
                    user_list: Some(users),
                    ..
                } => {
                    println!("{:<12} {:>4} {:>16}", "Username", "ID", "User type");
                    for user in users {
                        println!(
                            "{:<12} {:>4} {:>16}",
                            user.user_name,
                            user.user_id
                                .map(|id| id.to_string())
                                .unwrap_or_else(|| "-".to_owned()),
                            match user.user_level {
                                0 => "User".to_owned(),
                                1 => "Administrator".to_owned(),
                                n => format!("Unknown ({})", n),
                            },
                        );
                    }
                }
                _ => {
                    eprintln!("No users were included in response");
                    std::process::exit(1);
                }
            }
        }
        UserAction::Add {
            user_name,
            password,
            user_type,
        } => {
            camera
                .run_task(|cam| {
                    Box::pin({
                        let user_name = user_name.clone();
                        let password = password.clone();
                        let user_type = user_type.clone();
                        async move {
                            let user_list = cam
                                .get_users()
                                .await
                                .context("Failed to list current users")?;
                            let user_exists = match user_list.user_list {
                                Some(users) => users.iter().any(|user| user.user_name == user_name),
                                _ => false,
                            };
                            if user_exists {
                                return Err(anyhow!("The user '{}' already exists.", user_name));
                            }

                            cam.add_user(
                                user_name,
                                password,
                                match user_type {
                                    UserType::User => 0,
                                    UserType::Administrator => 1,
                                },
                            )
                            .await
                            .context("Unable to create user")
                        }
                    })
                })
                .await?;
            println!("Successfully created user '{}'", user_name);
        }
        UserAction::Password {
            user_name,
            password,
        } => {
            camera
                .run_task(|cam| {
                    Box::pin({
                        let user_name = user_name.clone();
                        let password = password.clone();
                        async move {
                            let user_list = cam
                                .get_users()
                                .await
                                .context("Failed to list current users")?;
                            let user_exists = match user_list.user_list {
                                Some(users) => users.iter().any(|user| user.user_name == user_name),
                                _ => false,
                            };
                            if !user_exists {
                                return Err(anyhow!("The user '{}' does not exist.", user_name));
                            }

                            cam.modify_user(user_name, password)
                                .await
                                .context("Unable to create user")
                        }
                    })
                })
                .await?;
            println!("Successfully changed the password of '{}'", user_name);
        }
        UserAction::Delete { user_name } => {
            camera
                .run_task(|cam| {
                    Box::pin({
                        let user_name = user_name.clone();
                        async move {
                            let user_list = cam
                                .get_users()
                                .await
                                .context("Failed to list current users")?;
                            let user_exists = match user_list.user_list {
                                Some(users) => users.iter().any(|user| user.user_name == user_name),
                                _ => false,
                            };

                            if !user_exists {
                                return Err(anyhow!("The user '{}' does not exist.", user_name));
                            }

                            cam.delete_user(user_name)
                                .await
                                .context("Unable to delete user")
                        }
                    })
                })
                .await?;
            println!("Successfully removed user '{}'", user_name);
        }
    }

    Ok(())
}
