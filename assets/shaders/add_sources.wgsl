struct AddSourcesUniforms {
    dt: f32,
    dimensions: vec2<f32>,
    cursor_position: vec2<f32>,
    previous_cursor_position: vec2<f32>,
    cursor_velocity: vec2<f32>,
    cursor_density: f32,
    density_source_active: u32,
    velocity_source_active: u32,
    density_stroke_continuous: u32,
    velocity_stroke_continuous: u32,
};

struct DensityVisualizeUniforms {
    color: vec4<f32>,
    fluid_intensity_scale: f32,
};

@group(0) @binding(0)
var density_in: texture_storage_2d<r32float, read>;

@group(0) @binding(1)
var velocity_in: texture_storage_2d<rg32float, read>;

@group(0) @binding(2)
var dye_in: texture_storage_2d<rgba16float, read>;

@group(0) @binding(3)
var density_out: texture_storage_2d<r32float, write>;

@group(0) @binding(4)
var velocity_out: texture_storage_2d<rg32float, write>;

@group(0) @binding(5)
var dye_out: texture_storage_2d<rgba16float, write>;

@group(0) @binding(6)
var<uniform> config: AddSourcesUniforms;

@group(0) @binding(7)
var<uniform> visualization: DensityVisualizeUniforms;

fn is_boundary(p: vec2<i32>) -> bool {
    let width = i32(config.dimensions.x);
    let height = i32(config.dimensions.y);
    return p.x == 0 || p.y == 0 || p.x == width - 1 || p.y == height - 1;
}

fn stroke_coverage(pixel: vec2<f32>, stroke_continuous: u32) -> f32 {
    let cursor = config.cursor_position * config.dimensions;
    let previous_cursor = select(
        cursor,
        config.previous_cursor_position * config.dimensions,
        stroke_continuous != 0u,
    );
    let segment = cursor - previous_cursor;
    let segment_length_squared = dot(segment, segment);
    let t = clamp(
        dot(pixel - previous_cursor, segment) / max(segment_length_squared, 0.000001),
        0.0,
        1.0,
    );
    let distance_to_stroke = length(pixel - (previous_cursor + t * segment));
    let radius = 0.0125 * min(config.dimensions.x, config.dimensions.y);
    let feather = 2.0;

    return 1.0 - smoothstep(radius - feather, radius + feather, distance_to_stroke);
}

@compute
@workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= u32(config.dimensions.x) || id.y >= u32(config.dimensions.y)) {
        return;
    }

    let p = vec2<i32>(id.xy);
    var density = textureLoad(density_in, p).r;
    var velocity = textureLoad(velocity_in, p).rg;
    var dye = textureLoad(dye_in, p).rgb;

    if (!is_boundary(p)) {
        let pixel = vec2<f32>(id.xy) + vec2<f32>(0.5);

        if (config.density_source_active != 0u) {
            let coverage = stroke_coverage(pixel, config.density_stroke_continuous);
            let added_density = coverage * config.cursor_density * config.dt;
            density += added_density;
            dye += visualization.color.rgb * added_density;
        }
        if (config.velocity_source_active != 0u) {
            const VELOCITY_SOURCE_STRENGTH: f32 = 10.0;
            let coverage = stroke_coverage(pixel, config.velocity_stroke_continuous);
            velocity += coverage
                * config.cursor_velocity
                * config.dt
                * VELOCITY_SOURCE_STRENGTH;
        }
    }

    textureStore(density_out, p, vec4<f32>(density, 0.0, 0.0, 0.0));
    textureStore(velocity_out, p, vec4<f32>(velocity, 0.0, 0.0));
    textureStore(dye_out, p, vec4<f32>(dye, 0.0));
}
