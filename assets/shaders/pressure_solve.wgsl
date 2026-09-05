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
var divergence: texture_storage_2d<r32float, read>;

@group(0) @binding(1)
var pressure_in: texture_storage_2d<r32float, read>;

@group(0) @binding(2)
var pressure_out: texture_storage_2d<r32float, write>;

@group(0) @binding(3)
var<uniform> config: FluidSimulationUniforms;

fn divergence_at(p: vec2<i32>) -> f32 {
    return textureLoad(divergence, p).r;
}

fn pressure_at(p: vec2<i32>) -> f32 {
    return textureLoad(pressure_in, p).r;
}

@compute
@workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (
        id.x >= u32(config.dimensions.x) ||
        id.y >= u32(config.dimensions.y)
    ) {
        return;
    }

    let p = vec2<i32>(id.xy);
    let width = i32(config.dimensions.x);
    let height = i32(config.dimensions.y);

    // Handle boundary conditiosn in a separate pass
    if (
        p.x == 0 ||
        p.x == width - 1 ||
        p.y == 0 ||
        p.y == height - 1
    ) {
        return;
    }

    let div = divergence_at(p);

    let left  = pressure_at(p + vec2<i32>(-1, 0));
    let right = pressure_at(p + vec2<i32>( 1, 0));
    let down  = pressure_at(p + vec2<i32>( 0,-1));
    let up    = pressure_at(p + vec2<i32>( 0, 1));

    let pressure =
        (div + left + right + down + up) / 4.0;

    textureStore(
        pressure_out,
        p,
        vec4<f32>(pressure, 0.0, 0.0, 0.0),
    );
}
