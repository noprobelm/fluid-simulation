struct AdvectionUniforms {
    dt: f32,
    dimensions: vec2<f32>,
};

@group(0) @binding(0)
var velocity_in: texture_2d<f32>;

@group(0) @binding(1)
var velocity_out: texture_storage_2d<rg32float, write>;

@group(0) @binding(2)
var<uniform> config: AdvectionUniforms;

fn advected_velocity_at(p: vec2<i32>) -> vec2<f32> {
    let n = max(config.dimensions - vec2<f32>(2.0), vec2<f32>(1.0));

    let velocity = velocity_at(p);

    var x = f32(p.x) - config.dt * n.x * velocity.x;
    var y = f32(p.y) - config.dt * n.y * velocity.y;

    x = clamp(x, 0.5, n.x + 0.5);
    y = clamp(y, 0.5, n.y + 0.5);

    let i0 = i32(floor(x));
    let i1 = i0 + 1;
    let j0 = i32(floor(y));
    let j1 = j0 + 1;

    let s1 = x - f32(i0);
    let s0 = 1.0 - s1;
    let t1 = y - f32(j0);
    let t0 = 1.0 - t1;

    let v00 = velocity_at(vec2<i32>(i0, j0));
    let v01 = velocity_at(vec2<i32>(i0, j1));
    let v10 = velocity_at(vec2<i32>(i1, j0));
    let v11 = velocity_at(vec2<i32>(i1, j1));

    return
          s0 * (t0 * v00 + t1 * v01)
        + s1 * (t0 * v10 + t1 * v11);
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

    let is_left = p.x == 0;
    let is_right = p.x == width - 1;
    let is_bottom = p.y == 0;
    let is_top = p.y == height - 1;

    let is_boundary =
        is_left ||
        is_right ||
        is_bottom ||
        is_top;

    // ------------------------------------------------------------
    // Interior
    // ------------------------------------------------------------

    if (!is_boundary) {
        let v = advected_velocity_at(p);

        textureStore(
            velocity_out,
            p,
            vec4<f32>(v.x, v.y, 0.0, 0.0),
        );

        return;
    }

    let interior_p = vec2<i32>(
        clamp(p.x, 1, width - 2),
        clamp(p.y, 1, height - 2),
    );

    var v = advected_velocity_at(interior_p);

    // At vertical walls, reverse the x component.
    if (is_left || is_right) {
        v.x = -v.x;
    }

    // At horizontal walls, reverse the y component.
    if (is_bottom || is_top) {
        v.y = -v.y;
    }

    textureStore(
        velocity_out,
        p,
        vec4<f32>(v.x, v.y, 0.0, 0.0),
    );
}
