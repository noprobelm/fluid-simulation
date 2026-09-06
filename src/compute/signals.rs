use bevy::{
    prelude::*,
    render::extract_resource::{ExtractResource, ExtractResourcePlugin},
};

pub(super) struct SignalsPlugin;

impl Plugin for SignalsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ResetSimulation>()
            .add_plugins(ExtractResourcePlugin::<ResetSimulation>::default());
    }
}

#[derive(Clone, Default, ExtractResource, Resource)]
pub struct ResetSimulation {
    pub generation: u64,
}
