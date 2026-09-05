struct FluidSimulationUniforms {
    alive_color: vec4<f32>,
    cursor_position: vec2<f32>,
    cursor_velocity: vec2<f32>,
    cursor_density: f32,
    time: f32,
    diff: f32,
    density_source_active: u32,
    velocity_source_active: u32,
    dimensions: vec2<f32>,
};

@group(0) @binding(0)
var pressure: texture_storage_2d<r32float, read>;

@group(0) @binding(1)
var velocity_in: texture_storage_2d<rg32float, read>;

@group(0) @binding(2)
var velocity_out: texture_storage_2d<rg32float, write>;

@group(0) @binding(3)
var<uniform> config: FluidSimulationUniforms;

@compute
@workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= u32(config.dimensions.x) || id.y >= u32(config.dimensions.y)) {
        return;
    }

    // Pressure projection is not implemented yet. Preserve the velocity field
    // instead of leaving the ping-pong output unwritten.
    let p = vec2<i32>(id.xy);
    let velocity = textureLoad(velocity_in, p).rg;
    textureStore(velocity_out, p, vec4<f32>(velocity, 0.0, 0.0));
}
