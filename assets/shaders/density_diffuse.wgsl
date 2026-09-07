struct DensityDiffuseUniforms {
    dt: f32,
    diff: f32,
    dimensions: vec2<f32>,
};

@group(0) @binding(0)
var density_original: texture_2d<f32>;

@group(0) @binding(1)
var dye_original: texture_2d<f32>;

@group(0) @binding(2)
var density_in: texture_2d<f32>;

@group(0) @binding(3)
var dye_in: texture_2d<f32>;

@group(0) @binding(4)
var density_out: texture_storage_2d<r32float, write>;

@group(0) @binding(5)
var dye_out: texture_storage_2d<rgba16float, write>;

@group(0) @binding(6)
var<uniform> config: DensityDiffuseUniforms;

fn density_at(p: vec2<i32>) -> f32 {
    return textureLoad(density_in, p, 0).r;
}

fn source_at(p: vec2<i32>) -> f32 {
    return textureLoad(density_original, p, 0).r;
}

fn dye_at(p: vec2<i32>) -> vec3<f32> {
    return textureLoad(dye_in, p, 0).rgb;
}

fn dye_source_at(p: vec2<i32>) -> vec3<f32> {
    return textureLoad(dye_original, p, 0).rgb;
}

fn diffuse_density_at(p: vec2<i32>) -> f32 {
    let n = max(config.dimensions.x - 2.0, 1.0);
    let a = config.dt * config.diff * n * n;

    let source = source_at(p);

    let left  = density_at(p + vec2<i32>(-1,  0));
    let right = density_at(p + vec2<i32>( 1,  0));
    let down  = density_at(p + vec2<i32>( 0, -1));
    let up    = density_at(p + vec2<i32>( 0,  1));

    return (
        source +
        a * (left + right + down + up)
    ) / (1.0 + 4.0 * a);
}

fn diffuse_dye_at(p: vec2<i32>) -> vec3<f32> {
    let n = max(config.dimensions.x - 2.0, 1.0);
    let a = config.dt * config.diff * n * n;

    let source = dye_source_at(p);
    let left  = dye_at(p + vec2<i32>(-1,  0));
    let right = dye_at(p + vec2<i32>( 1,  0));
    let down  = dye_at(p + vec2<i32>( 0, -1));
    let up    = dye_at(p + vec2<i32>( 0,  1));

    return (
        source +
        a * (left + right + down + up)
    ) / (1.0 + 4.0 * a);
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

    let is_boundary =
        p.x == 0 ||
        p.x == width - 1 ||
        p.y == 0 ||
        p.y == height - 1;

    if (!is_boundary) {
        let density = diffuse_density_at(p);
        let dye = diffuse_dye_at(p);

        textureStore(
            density_out,
            p,
            vec4<f32>(density, 0.0, 0.0, 0.0),
        );
        textureStore(dye_out, p, vec4<f32>(dye, 0.0));

        return;
    }

    let interior_p = vec2<i32>(
        clamp(p.x, 1, width - 2),
        clamp(p.y, 1, height - 2),
    );

    let density = diffuse_density_at(interior_p);
    let dye = diffuse_dye_at(interior_p);

    textureStore(
        density_out,
        p,
        vec4<f32>(density, 0.0, 0.0, 0.0),
    );
    textureStore(dye_out, p, vec4<f32>(dye, 0.0));
}
