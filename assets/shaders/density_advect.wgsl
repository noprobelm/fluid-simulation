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
var density_in: texture_storage_2d<r32float, read>;

@group(0) @binding(1)
var velocity_in: texture_storage_2d<rg32float, read>;

@group(0) @binding(2)
var density_out: texture_storage_2d<r32float, write>;

@group(0) @binding(3)
var<uniform> config: FluidSimulationUniforms;

fn density_at(p: vec2<i32>) -> f32 {
    return textureLoad(density_in, p).r;
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

    // Scalar boundary condition.
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

    let n_x = max(config.dimensions.x - 2.0, 1.0);
    let n_y = max(config.dimensions.y - 2.0, 1.0);
    let velocity = textureLoad(velocity_in, p).rg;

    var x = f32(p.x) - config.time * n_x * velocity.x;
    var y = f32(p.y) - config.time * n_y * velocity.y;

    x = clamp(x, 0.5, n_x + 0.5);
    y = clamp(y, 0.5, n_y + 0.5);

    let i0 = i32(floor(x));
    let i1 = i0 + 1;
    let j0 = i32(floor(y));
    let j1 = j0 + 1;

    let s1 = x - f32(i0);
    let s0 = 1.0 - s1;
    let t1 = y - f32(j0);
    let t0 = 1.0 - t1;

    let d00 = density_at(vec2<i32>(i0, j0));
    let d01 = density_at(vec2<i32>(i0, j1));
    let d10 = density_at(vec2<i32>(i1, j0));
    let d11 = density_at(vec2<i32>(i1, j1));

    let density = s0 * (t0 * d00 + t1 * d01)
                + s1 * (t0 * d10 + t1 * d11);

    textureStore(density_out, p, vec4<f32>(density, 0.0, 0.0, 0.0));
}


