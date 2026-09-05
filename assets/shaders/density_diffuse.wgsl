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
var density_original: texture_storage_2d<r32float, read>;

@group(0) @binding(1)
var density_in: texture_storage_2d<r32float, read>;

@group(0) @binding(2)
var density_out: texture_storage_2d<r32float, write>;

@group(0) @binding(3)
var<uniform> config: FluidSimulationUniforms;

fn density_at(p: vec2<i32>) -> f32 {
    return textureLoad(density_in, p).r;
}

fn source_at(p: vec2<i32>) -> f32 {
    return textureLoad(density_original, p).r;
}

fn diffuse_density_at(p: vec2<i32>) -> f32 {
    let n = max(config.dimensions - vec2<f32>(2.0), vec2<f32>(1.0));
    let a = config.time * config.diff * n * n;

    let source = source_at(p);

    let left  = density_at(p + vec2<i32>(-1,  0));
    let right = density_at(p + vec2<i32>( 1,  0));
    let down  = density_at(p + vec2<i32>( 0, -1));
    let up    = density_at(p + vec2<i32>( 0,  1));

    return (
        source +
        a.x * (left + right) + a.y * (down + up)
    ) / (1.0 + 2.0 * (a.x + a.y));
}
@compute
@workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= u32(config.dimensions.x) || id.y >= u32(config.dimensions.y)) {
        return;
    }

    let p = vec2<i32>(id.xy);
    let width = i32(config.dimensions.x);
    let height = i32(config.dimensions.y);

    // Scalar boundary condition from Stam's set_bnd(b = 0): copy the adjacent
    // interior value onto the outer boundary.
    if (p.x == 0) {
        textureStore(density_out, p, vec4<f32>(density_at(vec2<i32>(1, p.y)), 0.0, 0.0, 0.0));
        return;
    }
    if (p.x == width - 1) {
        textureStore(density_out, p, vec4<f32>(density_at(vec2<i32>(width - 2, p.y)), 0.0, 0.0, 0.0));
        return;
    }
    if (p.y == 0) {
        textureStore(density_out, p, vec4<f32>(density_at(vec2<i32>(p.x, 1)), 0.0, 0.0, 0.0));
        return;
    }
    if (p.y == height - 1) {
        textureStore(density_out, p, vec4<f32>(density_at(vec2<i32>(p.x, height - 2)), 0.0, 0.0, 0.0));
        return;
    }

    let n = max(config.dimensions - vec2<f32>(2.0), vec2<f32>(1.0));
    let a = config.time * config.diff * n * n;

    let source = textureLoad(density_original, p).r;
    let left = density_at(p + vec2<i32>(-1, 0));
    let right = density_at(p + vec2<i32>(1, 0));
    let down = density_at(p + vec2<i32>(0, -1));
    let up = density_at(p + vec2<i32>(0, 1));

    let density = (source + a.x * (left + right) + a.y * (down + up))
        / (1.0 + 2.0 * (a.x + a.y));

    textureStore(density_out, p, vec4<f32>(density, 0.0, 0.0, 0.0));
}

