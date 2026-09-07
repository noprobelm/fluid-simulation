mod signals;
mod states;

use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use signals::*;
pub use states::*;

use crate::compute::{
    AddSourcesUniforms, DensityDiffuseUniforms, DensityVisualizeUniforms, DisplayFactor,
    Iterations, ResetSimulation, VelocityDiffuseUniforms,
};

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
    mut sources: ResMut<AddSourcesUniforms>,
    mut vis: ResMut<DensityVisualizeUniforms>,
    mut density: ResMut<DensityDiffuseUniforms>,
    mut velocity: ResMut<VelocityDiffuseUniforms>,
    mut iterations: ResMut<Iterations>,
    mut display_factor: ResMut<DisplayFactor>,
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

            egui::CollapsingHeader::new("Sources")
                .default_open(true)
                .show(ui, |ui| {
                    egui::Grid::new("sources_grid")
                        .num_columns(2)
                        .spacing(egui::vec2(40.0, ui.spacing().item_spacing.y))
                        .show(ui, |ui| show_source_controls(ui, &mut sources));
                });

            egui::CollapsingHeader::new("Diffusion")
                .default_open(false)
                .show(ui, |ui| {
                    egui::Grid::new("diffusion_grid")
                        .num_columns(2)
                        .spacing(egui::vec2(40.0, ui.spacing().item_spacing.y))
                        .show(ui, |ui| {
                            show_diffusion_controls(ui, &mut iterations, &mut density)
                        });
                });

            egui::CollapsingHeader::new("Velocity")
                .default_open(false)
                .show(ui, |ui| {
                    egui::Grid::new("velocity_grid")
                        .num_columns(2)
                        .spacing(egui::vec2(40.0, ui.spacing().item_spacing.y))
                        .show(ui, |ui| show_velocity_controls(ui, &mut velocity));
                });

            egui::CollapsingHeader::new("Iterations")
                .default_open(false)
                .show(ui, |ui| {
                    egui::Grid::new("iterations_grid")
                        .num_columns(2)
                        .spacing(egui::vec2(40.0, ui.spacing().item_spacing.y))
                        .show(ui, |ui| show_pressure_controls(ui, &mut iterations));
                });

            let mut original = display_factor.0;
            egui::CollapsingHeader::new("Resolution")
                .default_open(false)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Display Factor");
                        ui.add(egui::Slider::new(&mut original, 1..=8));
                    });
                });
            display_factor.set_if_neq(DisplayFactor(original));
        });
    Ok(())
}

fn show_source_controls(ui: &mut egui::Ui, sources: &mut ResMut<AddSourcesUniforms>) {
    ui.label("Density Radius");
    ui.add(egui::Slider::new(&mut sources.density_radius, 0.001..=0.1));
    ui.end_row();

    ui.label("Velocity Radius");
    ui.add(egui::Slider::new(&mut sources.velocity_radius, 0.001..=0.1));
    ui.end_row();
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

fn show_pressure_controls(ui: &mut egui::Ui, iterations: &mut ResMut<Iterations>) {
    ui.label("Pressure Solve Iterations");
    ui.add(egui::Slider::new(&mut iterations.pressure_solve, 1..=20).step_by(0.1));
    ui.end_row();
}

fn show_diffusion_controls(
    ui: &mut egui::Ui,
    iterations: &mut ResMut<Iterations>,
    density: &mut ResMut<DensityDiffuseUniforms>,
) {
    ui.label("Mass Diffusion Coefficient");
    ui.add(
        egui::Slider::new(&mut density.diff, 1.0e-9..=1.0e-3)
            .logarithmic(true)
            .custom_formatter(|value, _| format!("{value:.1e}")),
    );
    ui.end_row();

    ui.label("Density Diffusion Iterations");
    ui.add(egui::Slider::new(&mut iterations.density_diffusion, 1..=20).step_by(0.1));
    ui.end_row();

    ui.label("Velocity Diffusion Iterations");
    ui.add(egui::Slider::new(&mut iterations.velocity_diffusion, 1..=20).step_by(0.1));
    ui.end_row();
}

fn show_velocity_controls(ui: &mut egui::Ui, velocity: &mut ResMut<VelocityDiffuseUniforms>) {
    ui.label("Kinematic Velocity");
    ui.add(
        egui::Slider::new(&mut velocity.visc, 1.0e-9..=1.0e-3)
            .logarithmic(true)
            .custom_formatter(|value, _| format!("{value:.1e}")),
    );
    ui.end_row();
}
