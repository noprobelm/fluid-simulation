use bevy::prelude::*;
use bevy_egui::EguiPrimaryContextPass;

use crate::{
    compute::{DisplayFactor, FluidSimUniforms},
    ui::UiState,
};

pub struct CursorPlugin;

impl Plugin for CursorPlugin {
    fn build(&self, app: &mut App) {
        app.init_state::<UiState>().add_systems(
            EguiPrimaryContextPass,
            update_cursor_input.run_if(in_state(UiState::Canvas)),
        );
    }
}

fn update_cursor_input(
    windows: Query<&Window>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    time: Res<Time>,
    display_factor: Res<DisplayFactor>,
    mut uniforms: ResMut<FluidSimUniforms>,
    mut previous_cursor_position: Local<Option<Vec2>>,
) -> Result {
    let window = windows.single()?;

    if let Some(cursor_position) = window.cursor_position() {
        let size = uniforms.dimensions.as_uvec2();
        let texture_size = size.as_vec2() * display_factor.0 as f32;
        let texture_origin = (window.size() - texture_size) * 0.5;
        let relative = (cursor_position - texture_origin) / texture_size;

        let cursor_position = relative.clamp(Vec2::ZERO, Vec2::ONE);
        let delta_seconds = time.delta().as_secs_f32();
        uniforms.cursor_velocity = previous_cursor_position
            .filter(|_| delta_seconds > 0.0)
            .map_or(Vec2::ZERO, |previous| {
                (cursor_position - previous) / delta_seconds
            });
        uniforms.cursor_position = cursor_position;
        *previous_cursor_position = Some(cursor_position);
    } else {
        uniforms.cursor_velocity = Vec2::ZERO;
        *previous_cursor_position = None;
    }

    uniforms.density_source_active = u32::from(mouse_buttons.pressed(MouseButton::Left));
    uniforms.velocity_source_active = u32::from(mouse_buttons.pressed(MouseButton::Right));

    Ok(())
}
