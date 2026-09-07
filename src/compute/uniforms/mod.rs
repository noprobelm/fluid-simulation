mod cursor;

use bevy::{
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_resource::ShaderType,
    },
    window::PrimaryWindow,
};

use crate::compute::DisplayFactor;

pub struct UniformsPlugin;

impl Plugin for UniformsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            cursor::CursorPlugin,
            ExtractResourcePlugin::<DimensionsUniforms>::default(),
            ExtractResourcePlugin::<AddSourcesUniforms>::default(),
            ExtractResourcePlugin::<AdvectionUniforms>::default(),
            ExtractResourcePlugin::<DensityDiffuseUniforms>::default(),
            ExtractResourcePlugin::<VelocityDiffuseUniforms>::default(),
            ExtractResourcePlugin::<DensityVisualizeUniforms>::default(),
        ))
        .add_systems(Startup, initialize_uniforms)
        .add_systems(Update, update_timestep);
    }
}

#[derive(Resource, Clone, Default, ExtractResource, ShaderType)]
pub struct AddSourcesUniforms {
    pub dt: f32,
    pub dimensions: Vec2,
    pub cursor_position: Vec2,
    pub previous_cursor_position: Vec2,
    pub cursor_velocity: Vec2,
    pub cursor_density: f32,
    pub density_radius: f32,
    pub velocity_radius: f32,
    pub density_source_active: u32,
    pub velocity_source_active: u32,
    pub density_stroke_continuous: u32,
    pub velocity_stroke_continuous: u32,
}

#[derive(Resource, Clone, ExtractResource, ShaderType)]
pub struct DimensionsUniforms {
    pub dimensions: Vec2,
}

#[derive(Resource, Clone, ExtractResource, ShaderType)]
pub struct AdvectionUniforms {
    pub dt: f32,
    pub dimensions: Vec2,
}

#[derive(Resource, Clone, ExtractResource, ShaderType)]
pub struct DensityDiffuseUniforms {
    pub dt: f32,
    pub diff: f32,
    pub dimensions: Vec2,
}

#[derive(Resource, Clone, ExtractResource, ShaderType)]
pub struct VelocityDiffuseUniforms {
    pub dt: f32,
    pub visc: f32,
    pub dimensions: Vec2,
}

#[derive(Resource, Clone, ExtractResource, ShaderType)]
pub struct DensityVisualizeUniforms {
    pub color: LinearRgba,
    pub fluid_intensity_scale: f32,
}

impl Default for DensityVisualizeUniforms {
    fn default() -> Self {
        Self {
            color: LinearRgba::rgb(0.35, 0.55, 0.95),
            fluid_intensity_scale: 10.,
        }
    }
}

fn initialize_uniforms(
    mut commands: Commands,
    windows: Query<&Window, With<PrimaryWindow>>,
    display_factor: Res<DisplayFactor>,
) -> Result {
    let dimensions = (windows.single()?.resolution.size().as_uvec2() / display_factor.0).as_vec2();

    commands.insert_resource(AddSourcesUniforms {
        dt: 0.0,
        dimensions,
        cursor_position: Vec2::ZERO,
        previous_cursor_position: Vec2::ZERO,
        cursor_velocity: Vec2::ZERO,
        cursor_density: 10.0,
        density_radius: 0.005,
        velocity_radius: 0.001,
        density_source_active: 0,
        velocity_source_active: 0,
        density_stroke_continuous: 0,
        velocity_stroke_continuous: 0,
    });
    commands.insert_resource(DimensionsUniforms { dimensions });
    commands.insert_resource(AdvectionUniforms {
        dt: 0.0,
        dimensions,
    });
    commands.insert_resource(DensityDiffuseUniforms {
        dt: 0.0,
        diff: 0.000001,
        dimensions,
    });
    commands.insert_resource(VelocityDiffuseUniforms {
        dt: 0.0,
        visc: 0.00000001,
        dimensions,
    });
    commands.init_resource::<DensityVisualizeUniforms>();

    Ok(())
}

fn update_timestep(
    time: Res<Time>,
    mut add_sources: ResMut<AddSourcesUniforms>,
    mut advection: ResMut<AdvectionUniforms>,
    mut density_diffuse: ResMut<DensityDiffuseUniforms>,
    mut velocity_diffuse: ResMut<VelocityDiffuseUniforms>,
) {
    let dt = time.delta().as_secs_f32();
    add_sources.dt = dt;
    advection.dt = dt;
    density_diffuse.dt = dt;
    velocity_diffuse.dt = dt;
}
