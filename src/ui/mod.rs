mod signals;
mod states;

use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use signals::*;
pub use states::*;

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

fn show(mut contexts: EguiContexts) -> Result {
    let ctx = contexts.ctx_mut()?;

    let title = "Settings";
    egui::Window::new(title)
        .constrain_to(ctx.content_rect())
        .show(ctx, |ui| {
            egui::Grid::new("settings_grid")
                .num_columns(2)
                .spacing(egui::vec2(40.0, ui.spacing().item_spacing.y))
        });
    Ok(())
}
