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
var divergence: texture_storage_2d<r32float, read>;

@group(0) @binding(1)
var pressure_in: texture_storage_2d<r32float, read>;

@group(0) @binding(2)
var pressure_out: texture_storage_2d<r32float, write>;

@group(0) @binding(3)
var<uniform> config: FluidSimulationUniforms;

@compute
@workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // if (id.x >= u32(config.dimensions.x) || id.y >= u32(config.dimensions.y)) {
    //     return;
    // }
    //
    // let p = vec2<i32>(id.xy);
    // let width = i32(config.dimensions.x);
    // let height = i32(config.dimensions.y);
    //
    //
    // // Set boundary values to 0
    // if (
    //     p.x == 0 ||
    //     p.x == width - 1 ||
    //     p.y == 0 ||
    //     p.y == height - 1
    // ) {
    //     textureStore(
    //         divergence,
    //         p,
    //         vec4<f32>(0.0, 0.0, 0.0, 0.0),
    //     );
    //     return;
    // }
    //
    // let n = max(config.dimensions.x - 2.0, 1.0);
    // let h = 1.0 / n;
    //
    // let left = velocity_at(p + vec2<i32>(-1, 0));
    // let right = velocity_at(p + vec2<i32>(1, 0));
    // let down = velocity_at(p + vec2<i32>(0, -1));
    // let up = velocity_at(p + vec2<i32>(0, 1));
    //
    // let div = -0.5 * h * (right.x - left.x + up.y - down.y);
    // textureStore(divergence, p, vec4<f32>(div, 0.0, 0.0, 0.0));
}
