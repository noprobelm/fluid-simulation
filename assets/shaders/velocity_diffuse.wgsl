struct VelocityDiffuseUniforms {
    dt: f32,
    visc: f32,
    dimensions: vec2<f32>,
};

@group(0) @binding(0)
var velocity_original: texture_storage_2d<rg32float, read>;

@group(0) @binding(1)
var velocity_in: texture_storage_2d<rg32float, read>;

@group(0) @binding(2)
var velocity_out: texture_storage_2d<rg32float, write>;

@group(0) @binding(3)
var<uniform> config: VelocityDiffuseUniforms;

fn velocity_at(p: vec2<i32>) -> vec2<f32> {
    return textureLoad(velocity_in, p).rg;
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

    // Scalar boundary condition from Stam's set_bnd
    if (p.x == 0) {
        let v = velocity_at(vec2<i32>(1, p.y));
        textureStore(
            velocity_out,
            p,
            vec4<f32>(-v.x, v.y, 0.0, 0.0),
        );
        return;
    }

    if (p.x == width - 1) {
        let v = velocity_at(vec2<i32>(width - 2, p.y));
        textureStore(
            velocity_out,
            p,
            vec4<f32>(-v.x, v.y, 0.0, 0.0),
        );
        return;
    }

    if (p.y == 0) {
        let v = velocity_at(vec2<i32>(p.x, 1));
        textureStore(
            velocity_out,
            p,
            vec4<f32>(v.x, -v.y, 0.0, 0.0),
        );
        return;
    }

if (p.y == height - 1) {
    let v = velocity_at(vec2<i32>(p.x, height - 2));
    textureStore(
        velocity_out,
        p,
        vec4<f32>(v.x, -v.y, 0.0, 0.0),
    );
    return;
}

    let n = max(config.dimensions - vec2<f32>(2.0), vec2<f32>(1.0));
    let a = config.dt * config.visc * n * n;

    let source = textureLoad(velocity_original, p).rg;
    let left = velocity_at(p + vec2<i32>(-1, 0));
    let right = velocity_at(p + vec2<i32>(1, 0));
    let down = velocity_at(p + vec2<i32>(0, -1));
    let up = velocity_at(p + vec2<i32>(0, 1));

    let velocity = (source + a.x * (left + right) + a.y * (down + up))
        / (1.0 + 2.0 * (a.x + a.y));

    textureStore(velocity_out, p, vec4<f32>(velocity, 0.0, 0.0));
}
