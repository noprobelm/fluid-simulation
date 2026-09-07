//! This project defines and runs a compute shader to provide a real-time fluid dynamics simulation.
//! The implementation is based on a research paper written by Jos Stam: https://graphics.cs.cmu.edu/nsp/course/15-464/Fall09/papers/StamFluidforGames.pdf
//!
//! Our implementation here differs slightly from Stam's in that we are targeting the GPU for the
//! simulation instead of their CPU version.
//!
//! As such, we use textures to manage density, velocity, pressure, and divergence inputs. We also
//! use Jacobi iteration for diffusion and pressure solutions, as this felt more conducive to
//! compute shader architecture than Stam's Gauss-Seidel relaxation approach.
mod compute;
mod ui;

use bevy::{
    asset::{AssetMetaCheck, AssetPlugin},
    prelude::*,
    window::WindowMode,
};

use crate::{compute::ComputePlugin, ui::UiPlugin};

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .add_plugins((
            DefaultPlugins
                .set(AssetPlugin {
                    meta_check: AssetMetaCheck::Never,
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        mode: initial_window_mode(),
                        fit_canvas_to_parent: true,
                        ..default()
                    }),
                    ..default()
                })
                .set(ImagePlugin::default_nearest()),
            ComputePlugin,
            UiPlugin,
        ))
        .run();
}

fn initial_window_mode() -> WindowMode {
    #[cfg(target_arch = "wasm32")]
    return WindowMode::Windowed;

    #[cfg(not(target_arch = "wasm32"))]
    return WindowMode::BorderlessFullscreen(MonitorSelection::Primary);
}
