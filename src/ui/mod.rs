mod signals;
mod states;

use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use signals::*;
pub use states::*;

use crate::compute::DensityVisualizeUniforms;

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

fn show(mut contexts: EguiContexts, mut fluid_color: ResMut<DensityVisualizeUniforms>) -> Result {
    let ctx = contexts.ctx_mut()?;

    let title = "Settings";
    egui::Window::new(title)
        .constrain_to(ctx.content_rect())
        .show(ctx, |ui| {
            egui::Grid::new("settings_grid")
                .num_columns(2)
                .spacing(egui::vec2(40.0, ui.spacing().item_spacing.y))
                .show(ui, |ui| {
                    ui.heading("Visuals");
                    ui.end_row();
                    add_major_grid_separator(ui);

                    ui.label("Fluid Color");
                    let original = [
                        fluid_color.color.red,
                        fluid_color.color.green,
                        fluid_color.color.blue,
                    ];
                    let mut color32 = original;
                    ui.color_edit_button_rgb(&mut color32);

                    if original != color32 {
                        fluid_color.color = LinearRgba::new(
                            *color32.first().unwrap(),
                            *color32.get(1).unwrap(),
                            *color32.last().unwrap(),
                            1.0,
                        );
                    }
                });
        });
    Ok(())
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
