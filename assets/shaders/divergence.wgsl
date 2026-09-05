struct FluidSimulationUniforms {
    alive_color: vec4<f32>,
    cursor_position: vec2<f32>,
    cursor_density: f32,
    time: f32,
    diff: f32,
    dimensions: vec2<f32>,
};

@group(0) @binding(0)
var velocity_in: texture_storage_2d<rg32float, read>;

@group(0) @binding(1)
var divergence: texture_storage_2d<r32float, read_write>;

@group(0) @binding(2)
var<uniform> config: FluidSimulationUniforms;

fn divergence_at(p: vec2<i32>) -> f32 {
    return textureLoad(divergence, p).r;
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
        textureStore(divergence, p, vec4<f32>(divergence_at(vec2<i32>(1, p.y)), 0.0, 0.0, 0.0));
        return;
    }
    if (p.x == width - 1) {
        textureStore(divergence, p, vec4<f32>(divergence_at(vec2<i32>(width - 2, p.y)), 0.0, 0.0, 0.0));
        return;
    }
    if (p.y == 0) {
        textureStore(divergence, p, vec4<f32>(divergence_at(vec2<i32>(p.x, 1)), 0.0, 0.0, 0.0));
        return;
    }
    if (p.y == height - 1) {
        textureStore(divergence, p, vec4<f32>(divergence_at(vec2<i32>(p.x, height - 2)), 0.0, 0.0, 0.0));
        return;
    }

    let n = max(config.dimensions.x - 2.0, 1.0);
    let h = 1.0 / n;
    // let source = textureLoad(divergence, p).r;
    // let a = config.time * config.diff * n * n;
    //
    // let source = textureLoad(density_original, p).r;
    // let left = divergence_at(p + vec2<i32>(-1, 0));
    // let right = divergence_at(p + vec2<i32>(1, 0));
    // let down = divergence_at(p + vec2<i32>(0, -1));
    // let up = divergence_at(p + vec2<i32>(0, 1));
    //
    // let density = (source + a * (left + right + down + up)) / (1.0 + 4.0 * a);
    //
    // textureStore(divergence, p, vec4<f32>(density, 0.0, 0.0, 0.0));
}
