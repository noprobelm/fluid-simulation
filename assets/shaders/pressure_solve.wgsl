struct DimensionsUniforms {
    dimensions: vec2<f32>,
};

@group(0) @binding(0)
var divergence: texture_storage_2d<r32float, read>;

@group(0) @binding(1)
var pressure_in: texture_storage_2d<r32float, read>;

@group(0) @binding(2)
var pressure_out: texture_storage_2d<r32float, write>;

@group(0) @binding(3)
var<uniform> config: DimensionsUniforms;

fn divergence_at(p: vec2<i32>) -> f32 {
    return textureLoad(divergence, p).r;
}

fn pressure_at(p: vec2<i32>) -> f32 {
    return textureLoad(pressure_in, p).r;
}

fn solve_pressure_at(p: vec2<i32>) -> f32 {
    let n = max(config.dimensions - vec2<f32>(2.0), vec2<f32>(1.0));
    let inverse_spacing_squared = n * n;
    let div = divergence_at(p);

    let left  = pressure_at(p + vec2<i32>(-1,  0));
    let right = pressure_at(p + vec2<i32>( 1,  0));
    let down  = pressure_at(p + vec2<i32>( 0, -1));
    let up    = pressure_at(p + vec2<i32>( 0,  1));

    return (
        div +
        inverse_spacing_squared.x * (left + right) +
        inverse_spacing_squared.y * (down + up)
    ) / (2.0 * (inverse_spacing_squared.x + inverse_spacing_squared.y));
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

    if (is_boundary) {
        let interior_p = vec2<i32>(
            clamp(p.x, 1, width - 2),
            clamp(p.y, 1, height - 2),
        );

        // Equivalent to applying set_bnd(b = 0) from Stam's paper
        let value = solve_pressure_at(interior_p);

        textureStore(
            pressure_out,
            p,
            vec4<f32>(value, 0.0, 0.0, 0.0),
        );

        return;
    }

    let pressure = solve_pressure_at(p);

    textureStore(
        pressure_out,
        p,
        vec4<f32>(pressure, 0.0, 0.0, 0.0),
    );
}

