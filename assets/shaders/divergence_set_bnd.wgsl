struct FluidSimulationUniforms {
    color: vec4<f32>,
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
var velocity_in: texture_storage_2d<rg32float, read>;

@group(0) @binding(1)
var divergence: texture_storage_2d<r32float, write>;

@group(0) @binding(2)
var<uniform> config: FluidSimulationUniforms;

fn velocity_at(p: vec2<i32>) -> vec2<f32> {
    return textureLoad(velocity_in, p).rg;
}

fn divergence_at_interior(p: vec2<i32>) -> f32 {
    let n = max(config.dimensions - vec2<f32>(2.0), vec2<f32>(1.0));

    let left  = velocity_at(p + vec2<i32>(-1,  0));
    let right = velocity_at(p + vec2<i32>( 1,  0));
    let down  = velocity_at(p + vec2<i32>( 0, -1));
    let up    = velocity_at(p + vec2<i32>( 0,  1));

    return -0.5 * (
        n.x * (right.x - left.x) +
        n.y * (up.y - down.y)
    );
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

    let is_left   = p.x == 0;
    let is_right  = p.x == width - 1;
    let is_bottom = p.y == 0;
    let is_top    = p.y == height - 1;

    if !(is_left || is_right || is_bottom || is_top) {
        return;
    }

    let interior_p = vec2<i32>(
        clamp(p.x, 1, width - 2),
        clamp(p.y, 1, height - 2),
    );

    let div = divergence_at_interior(interior_p);

    textureStore(
        divergence,
        p,
        vec4<f32>(div, 0.0, 0.0, 0.0),
    );
}
