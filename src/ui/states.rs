use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPrimaryContextPass};

pub(super) struct StatesPlugin;

impl Plugin for StatesPlugin {
    fn build(&self, app: &mut App) {
        app.init_state::<UiState>()
            .add_systems(EguiPrimaryContextPass, handle_ui_state);
    }
}

#[derive(States, Reflect, Default, Debug, Clone, Eq, PartialEq, Hash)]
pub enum UiState {
    #[default]
    Canvas,
    Menu,
}

fn handle_ui_state(
    mut contexts: EguiContexts,
    current_ui_state: Res<State<UiState>>,
    mut next_ui_state: ResMut<NextState<UiState>>,
) -> Result {
    let ctx = contexts.ctx_mut()?;

    let is_pointer_over_area = ctx.is_pointer_over_egui();
    let is_using_pointer = ctx.egui_is_using_pointer();
    let wants_keyboard_input = ctx.egui_wants_keyboard_input();

    let should_be_ui = is_using_pointer || wants_keyboard_input || is_pointer_over_area;

    match current_ui_state.get() {
        UiState::Canvas => {
            if should_be_ui {
                next_ui_state.set(UiState::Menu);
            }
        }
        UiState::Menu => {
            if !should_be_ui {
                next_ui_state.set(UiState::Canvas);
            }
        }
    }

    Ok(())
}
