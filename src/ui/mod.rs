mod signals;
mod states;

use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use signals::*;
pub use states::*;

use crate::compute::{DensityVisualizeUniforms, ResetSimulation};

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

            add_major_grid_separator(ui);

            egui::Grid::new("settings_grid")
                .num_columns(2)
                .spacing(egui::vec2(40.0, ui.spacing().item_spacing.y))
                .show(ui, |ui| {
                    ui.heading("Visuals");
                    ui.end_row();
                    add_major_grid_separator(ui);

                    show_vis(ui, &mut vis);
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
}

pub fn add_major_grid_separator(ui: &mut egui::Ui) {
    let row_spacing = ui.spacing().item_spacing.y;
    let padding = 8.0;
    let rect = ui.max_rect();
    // Line is drawn at: current position + padding above
    // Total space needed: padding above + padding below - row_spacing (which end_row adds)
    let y = ui.cursor().top() + padding;
    let mut stroke = ui.visuals().widgets.noninteractive.bg_stroke;
    stroke.width = 2.0;
    ui.painter().hline(rect.left()..=rect.right(), y, stroke);
    ui.allocate_space(egui::vec2(0.0, (padding * 2.0 - row_spacing).max(0.0)));
    ui.end_row();
}
