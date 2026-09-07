struct DimensionsUniforms {
    dimensions: vec2<f32>,
};


@group(0) @binding(0)
var pressure: texture_2d<f32>;

@group(0) @binding(1)
var velocity_in: texture_2d<f32>;

@group(0) @binding(2)
var velocity_out: texture_storage_2d<rg32float, write>;

@group(0) @binding(3)
var<uniform> config: DimensionsUniforms;

fn projected_velocity_at(p: vec2<i32>) -> vec2<f32> {
    let n = max(config.dimensions - vec2<f32>(2.0), vec2<f32>(1.0));

    let left  = pressure_at(p + vec2<i32>(-1, 0));
    let right = pressure_at(p + vec2<i32>( 1, 0));
    let down  = pressure_at(p + vec2<i32>( 0,-1));
    let up    = pressure_at(p + vec2<i32>( 0, 1));

    let velocity = velocity_at(p);

    return vec2<f32>(
        velocity.x - 0.5 * n.x * (right - left),
        velocity.y - 0.5 * n.y * (up - down),
    );
}

fn pressure_at(p: vec2<i32>) -> f32 {
    return textureLoad(pressure, p, 0).r;
}

fn velocity_at(p: vec2<i32>) -> vec2<f32> {
    return textureLoad(velocity_in, p, 0).rg;
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

    if (!(is_left || is_right || is_bottom || is_top)) {
        let v = projected_velocity_at(p);

        textureStore(
            velocity_out,
            p,
            vec4<f32>(v, 0.0, 0.0),
        );

        return;
    }

    let interior_p = vec2<i32>(
        clamp(p.x, 1, width - 2),
        clamp(p.y, 1, height - 2),
    );

    let interior_v = projected_velocity_at(interior_p);

    var boundary_v = interior_v;

    if (is_left || is_right) {
        boundary_v.x = -boundary_v.x;
    }

    if (is_bottom || is_top) {
        boundary_v.y = -boundary_v.y;
    }

    textureStore(
        velocity_out,
        p,
        vec4<f32>(boundary_v, 0.0, 0.0),
    );
}
