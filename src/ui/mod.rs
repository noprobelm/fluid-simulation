mod signals;
mod states;

use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use signals::*;
pub use states::*;

use crate::compute::{DensityVisualizeUniforms, Iterations, ResetSimulation};

pub(super) struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((EguiPlugin::default(), StatesPlugin, SignalsPlugin))
            .add_systems(
                EguiPrimaryContextPass,
                show.run_if(resource_exists::<ShowUi>),
            )
            .init_resource::<ShowUi>();
    }
}

#[derive(Resource, Default)]
struct ShowUi;

fn show(
    mut contexts: EguiContexts,
    mut vis: ResMut<DensityVisualizeUniforms>,
    mut iterations: ResMut<Iterations>,
    mut reset: ResMut<ResetSimulation>,
) -> Result {
    let ctx = contexts.ctx_mut()?;

    let title = "Settings";
    egui::Window::new(title)
        .constrain_to(ctx.content_rect())
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Commands");
                if ui.button("Reset").clicked() {
                    reset.generation = reset.generation.wrapping_add(1);
                }
            });

            egui::CollapsingHeader::new("Visuals")
                .default_open(true)
                .show(ui, |ui| {
                    egui::Grid::new("visuals_grid")
                        .num_columns(2)
                        .spacing(egui::vec2(40.0, ui.spacing().item_spacing.y))
                        .show(ui, |ui| show_vis(ui, &mut vis));
                });

            egui::CollapsingHeader::new("Iterations")
                .default_open(false)
                .show(ui, |ui| {
                    egui::Grid::new("iterations_grid")
                        .num_columns(2)
                        .spacing(egui::vec2(40.0, ui.spacing().item_spacing.y))
                        .show(ui, |ui| show_simulation_controls(ui, &mut iterations));
                });
        });
    Ok(())
}

fn show_vis(ui: &mut egui::Ui, vis: &mut ResMut<DensityVisualizeUniforms>) {
    ui.label("Fluid Color");
    let original = [vis.color.red, vis.color.green, vis.color.blue];
    let mut color32 = original;
    ui.color_edit_button_rgb(&mut color32);

    if original != color32 {
        vis.color = LinearRgba::new(
            *color32.first().unwrap(),
            *color32.get(1).unwrap(),
            *color32.last().unwrap(),
            1.0,
        );
    }

    ui.end_row();

    ui.label("Intensity Scale");
    ui.add(egui::Slider::new(&mut vis.fluid_intensity_scale, 0.0..=20.0).step_by(0.1));
    ui.end_row();
}

fn show_simulation_controls(ui: &mut egui::Ui, iterations: &mut ResMut<Iterations>) {
    ui.label("Density Diffusion Iterations");
    ui.add(egui::Slider::new(&mut iterations.density_diffusion, 1..=20).step_by(0.1));
    ui.end_row();

    ui.label("Velocity Diffusion Iterations");
    ui.add(egui::Slider::new(&mut iterations.velocity_diffusion, 1..=20).step_by(0.1));
    ui.end_row();

    ui.label("Pressure Solve Iterations");
    ui.add(egui::Slider::new(&mut iterations.pressure_solve, 1..=20).step_by(0.1));
    ui.end_row();
}
