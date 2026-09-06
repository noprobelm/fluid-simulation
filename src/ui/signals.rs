use crate::ui::ShowUi;
use bevy::prelude::*;

pub(super) struct SignalsPlugin;

impl Plugin for SignalsPlugin {
    fn build(&self, app: &mut App) {
        app.add_observer(on_ui_toggle);
    }
}

#[derive(Event)]
pub struct UiToggleEvent;

fn on_ui_toggle(_trigger: On<UiToggleEvent>, mut commands: Commands, enabled: Option<Res<ShowUi>>) {
    if enabled.is_some() {
        commands.remove_resource::<ShowUi>();
    } else {
        commands.insert_resource(ShowUi);
    }
}
