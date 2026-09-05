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
var velocity_out: texture_storage_2d<rg32float, write>;

@group(0) @binding(4)
var<uniform> config: FluidSimulationUniforms;

fn is_boundary(p: vec2<i32>) -> bool {
    let width = i32(config.dimensions.x);
    let height = i32(config.dimensions.y);
    return p.x == 0 || p.y == 0 || p.x == width - 1 || p.y == height - 1;
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

    if (!is_boundary(p)) {
        let uv = (vec2<f32>(id.xy) + vec2<f32>(0.5)) / config.dimensions;
        let cursor = vec2<f32>(config.cursor_position.x, config.cursor_position.y);
        let radius = 0.0125;

        if (config.density_source_active != 0u && distance(uv, cursor) <= radius) {
            density += config.cursor_density * config.time;
        }
        if (config.velocity_source_active != 0u && distance(uv, cursor) <= radius) {
            velocity += config.cursor_velocity * config.time;
        }
    }

    textureStore(density_out, p, vec4<f32>(density, 0.0, 0.0, 0.0));
    textureStore(velocity_out, p, vec4<f32>(velocity, 0.0, 0.0));
}
