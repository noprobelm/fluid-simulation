// The shader reads the previous frame's state from the `input` texture, and writes the new state of
// each pixel to the `output` texture. The textures are flipped each step to progress the
// simulation.
// Two textures are needed for the game of life as each pixel of step N depends on the state of its
// neighbors at step N-1.

@group(0) @binding(0) var original: texture_storage_2d<rgba32float, read>;

@group(0) @binding(1) var current: texture_storage_2d<rgba32float, read>;

@group(0) @binding(2) var next: texture_storage_2d<rgba32float, write>;


@group(0) @binding(3) var<uniform> config: FluidSimulationUniforms;

struct FluidSimulationUniforms {
    alive_color: vec4<f32>,
    cursor_position: vec2<f32>,
    cursor_density: f32,
    time: f32,
    diff: f32,
    dimensions: vec2<f32>,
}

fn hash(value: u32) -> u32 {
    var state = value;
    state = state ^ 2747636419u;
    state = state * 2654435769u;
    state = state ^ (state >> 16u);
    state = state * 2654435769u;
    state = state ^ (state >> 16u);
    state = state * 2654435769u;
    return state;
}

fn randomFloat(value: u32) -> f32 {
    return f32(hash(value)) / 4294967295.0;
}

@compute @workgroup_size(8, 8, 1)
fn init(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    if (invocation_id.x >= u32(config.dimensions.x) ||
        invocation_id.y >= u32(config.dimensions.y)) {
        return;
    }

    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

    let randomNumber = randomFloat((invocation_id.y << 16u) | invocation_id.x);
    let alive = randomNumber > 0.9;
    // Use alpha channel to keep track of cell's state
    let color = vec4(0., 0., 0., 1);

    textureStore(next, location, color);
}

fn is_alive(location: vec2<i32>, offset_x: i32, offset_y: i32) -> i32 {
    let value: vec4<f32> = textureLoad(current, location + vec2<i32>(offset_x, offset_y));
    return i32(value.a);
}

fn count_alive(location: vec2<i32>) -> i32 {
    return is_alive(location, -1, -1) +
           is_alive(location, -1,  0) +
           is_alive(location, -1,  1) +
           is_alive(location,  0, -1) +
           is_alive(location,  0,  1) +
           is_alive(location,  1, -1) +
           is_alive(location,  1,  0) +
           is_alive(location,  1,  1);
}

fn density_source_at(location: vec2<i32>) -> f32 {
    let cursor = config.cursor_position * config.dimensions;
    let p = vec2<f32>(location);

    let distance = length(p - cursor);

    let radius = 5.0;

    let strength = 1.0 - clamp(
        distance / radius,
        0.0,
        1.0
    );

    return config.cursor_density * strength;
}

fn set_bnd(location: vec2<i32>, value: vec4<f32>, b: i32) -> vec4<f32> {
    let width = i32(config.dimensions.x);
    let height = i32(config.dimensions.y);

    var result = value;

    // Left boundary
    if (location.x == 0) {
        let neighbor = textureLoad(current, vec2<i32>(1, location.y));

        result.r = select(
            neighbor.r,
            -neighbor.r,
            b == 1
        );
    }

    // Right boundary
    else if (location.x == width - 1) {
        let neighbor = textureLoad(
            current,
            vec2<i32>(width - 2, location.y)
        );

        result.r = select(
            neighbor.r,
            -neighbor.r,
            b == 1
        );
    }

    // Bottom boundary
    else if (location.y == 0) {
        let neighbor = textureLoad(current, vec2<i32>(location.x, 1));

        result.r = select(
            neighbor.r,
            -neighbor.r,
            b == 2
        );
    }

    // Top boundary
    else if (location.y == height - 1) {
        let neighbor = textureLoad(
            current,
            vec2<i32>(location.x, height - 2)
        );

        result.r = select(
            neighbor.r,
            -neighbor.r,
            b == 2
        );
    }

    return result;
}

@compute @workgroup_size(8, 8, 1)
fn diffuse(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
  // JAB TODO: Why is this check necessary
  if (invocation_id.x >= u32(config.dimensions.x) ||
      invocation_id.y >= u32(config.dimensions.y)) {
      return;
  }

  let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

  let width = i32(config.dimensions.x);
  let height = i32(config.dimensions.y);

  if location.x == 0 ||
     location.y == 0 ||
     location.x == width - 1 ||
     location.y == height - 1 {
     return;
  }

  let n = config.dimensions.x - 2.0;
  let a = config.time * config.diff * n * n;

  var x0 = textureLoad(original, location);

  let left = textureLoad(current, location + vec2<i32>(-1, 0));
  let right = textureLoad(current, location + vec2<i32>(1, 0));
  let below = textureLoad(current, location + vec2<i32>(0, -1));
  let above = textureLoad(current, location + vec2<i32>(0, 1));

  var result = textureLoad(current, location);

  result.r = (
      x0.r +
      a * (
          left.r +
          right.r +
          below.r +
          above.r
      )
  ) / (1.0 + 4.0 * a);

  textureStore(next, location, result);
}

@compute @workgroup_size(8, 8, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    if (invocation_id.x >= u32(config.dimensions.x) ||
        invocation_id.y >= u32(config.dimensions.y)) {
        return;
    }

    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    var color = textureLoad(current, location);
    color.r += config.time * density_source_at(location);
    color = set_bnd(location, color, 0);
    textureStore(next, location, color);
}
