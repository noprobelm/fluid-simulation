use bevy::prelude::*;
use bevy_egui::EguiPrimaryContextPass;

use crate::{compute::DisplayFactor, ui::UiState};

use super::AddSourcesUniforms;

pub struct CursorPlugin;

impl Plugin for CursorPlugin {
    fn build(&self, app: &mut App) {
        app.init_state::<UiState>()
            .add_systems(EguiPrimaryContextPass, update_cursor_input);
    }
}

#[derive(Default)]
struct CursorHistory {
    pointer: Option<Pointer>,
    position: Option<Vec2>,
    density_active: bool,
    velocity_active: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Pointer {
    Mouse,
    Touch(u64),
}

fn update_cursor_input(
    windows: Query<&Window>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    touches: Res<Touches>,
    time: Res<Time>,
    ui_state: Res<State<UiState>>,
    display_factor: Res<DisplayFactor>,
    mut uniforms: ResMut<AddSourcesUniforms>,
    mut history: Local<CursorHistory>,
) -> Result {
    let window = windows.single()?;
    let touch = touches.iter().min_by_key(|touch| touch.id());
    let (pointer, position, density_active, velocity_active) = if let Some(touch) = touch {
        (
            Pointer::Touch(touch.id()),
            Some(touch.position()),
            true,
            true,
        )
    } else {
        (
            Pointer::Mouse,
            window.cursor_position(),
            mouse_buttons.pressed(MouseButton::Left),
            mouse_buttons.pressed(MouseButton::Right),
        )
    };

    let cursor_position = position.and_then(|position| {
        let size = uniforms.dimensions.as_uvec2();
        let texture_size = size.as_vec2() * display_factor.0 as f32;
        let texture_origin = (window.size() - texture_size) * 0.5;
        let relative = (position - texture_origin) / texture_size;

        ((0.0..=1.0).contains(&relative.x) && (0.0..=1.0).contains(&relative.y)).then_some(relative)
    });

    let input_enabled = matches!(ui_state.get(), UiState::Canvas);
    let cursor_position = cursor_position.filter(|_| input_enabled);

    if cursor_position.is_some() && history.pointer != Some(pointer) {
        *history = CursorHistory {
            pointer: Some(pointer),
            ..default()
        };
    }

    update_source_input(
        &mut uniforms,
        &mut history,
        cursor_position,
        density_active,
        velocity_active,
        time.delta().as_secs_f32(),
    );

    Ok(())
}

fn update_source_input(
    uniforms: &mut AddSourcesUniforms,
    history: &mut CursorHistory,
    cursor_position: Option<Vec2>,
    density_active: bool,
    velocity_active: bool,
    dt: f32,
) {
    let Some(cursor_position) = cursor_position else {
        uniforms.cursor_velocity = Vec2::ZERO;
        uniforms.density_source_active = 0;
        uniforms.velocity_source_active = 0;
        uniforms.density_stroke_continuous = 0;
        uniforms.velocity_stroke_continuous = 0;
        *history = CursorHistory::default();
        return;
    };

    uniforms.previous_cursor_position = history.position.unwrap_or(cursor_position);
    uniforms.cursor_position = cursor_position;
    uniforms.cursor_velocity = history
        .position
        .filter(|_| dt > 0.0)
        .map_or(Vec2::ZERO, |previous| (cursor_position - previous) / dt);
    uniforms.density_source_active = u32::from(density_active);
    uniforms.velocity_source_active = u32::from(velocity_active);
    uniforms.density_stroke_continuous = u32::from(density_active && history.density_active);
    uniforms.velocity_stroke_continuous = u32::from(velocity_active && history.velocity_active);

    history.position = Some(cursor_position);
    history.density_active = density_active;
    history.velocity_active = velocity_active;
}
