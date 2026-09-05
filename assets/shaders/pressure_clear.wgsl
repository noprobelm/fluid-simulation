struct FluidSimulationUniforms {
    alive_color: vec4<f32>,
    cursor_position: vec2<f32>,
    cursor_velocity: vec2<f32>,
    cursor_density: f32,
    time: f32,
    diff: f32,
    visc: f32,
    density_source_active: u32,
    velocity_source_active: u32,
    dimensions: vec2<f32>,
};

@group(0) @binding(0)
var pressure_out: texture_storage_2d<r32float, write>;

@group(0) @binding(1)
var<uniform> config: FluidSimulationUniforms;

@compute
@workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (
        id.x >= u32(config.dimensions.x) ||
        id.y >= u32(config.dimensions.y)
    ) {
        return;
    }

    textureStore(
        pressure_out,
        vec2<i32>(id.xy),
        vec4<f32>(0.0, 0.0, 0.0, 0.0),
    );
}
