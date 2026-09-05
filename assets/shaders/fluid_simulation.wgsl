@group(0) @binding(0) var density_original: texture_storage_2d<r32float, read>;

@group(0) @binding(1) var density_current: texture_storage_2d<r32float, read>;

@group(0) @binding(2) var density_next: texture_storage_2d<r32float, write>;

@group(0) @binding(3) var velocity_current: texture_storage_2d<rg32float, read>;

@group(0) @binding(4) var velocity_next: texture_storage_2d<rg32float, write>;

@group(0) @binding(5) var<uniform> config: FluidSimulationUniforms;

struct FluidSimulationUniforms {
    alive_color: vec4<f32>,
    cursor_position: vec2<f32>,
    cursor_density: f32,
    time: f32,
    diff: f32,
    dimensions: vec2<f32>,
}

@compute @workgroup_size(8, 8, 1)
fn init(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    if (invocation_id.x >= u32(config.dimensions.x) ||
        invocation_id.y >= u32(config.dimensions.y)) {
        return;
    }

    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

    let color = vec4(0., 0., 0., 1);

    textureStore(density_next, location, color);
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
        let neighbor = textureLoad(density_current, vec2<i32>(1, location.y));

        result.r = select(
            neighbor.r,
            -neighbor.r,
            b == 1
        );
    }

    // Right boundary
    else if (location.x == width - 1) {
        let neighbor = textureLoad(
            density_current,
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
        let neighbor = textureLoad(density_current, vec2<i32>(location.x, 1));

        result.r = select(
            neighbor.r,
            -neighbor.r,
            b == 2
        );
    }

    // Top boundary
    else if (location.y == height - 1) {
        let neighbor = textureLoad(
            density_current,
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
fn advect(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
  if (invocation_id.x >= u32(config.dimensions.x) ||
        invocation_id.y >= u32(config.dimensions.y)) {
        return;
    }

    let location = vec2<i32>(
        i32(invocation_id.x),
        i32(invocation_id.y)
    );

    let n = config.dimensions.x - 2.0;
    let dt0 = config.time * n;

    let velocity = textureLoad(velocity_current, location);

    var x = f32(location.x) - dt0 * velocity.x;
    var y = f32(location.y) - dt0 * velocity.y;

    x = clamp(x, 0.5, n + 0.5);
    y = clamp(y, 0.5, n + 0.5);

    let i0 = i32(x);
    let i1 = i0 + 1;
    let j0 = i32(y);
    let j1 = j0 + 1;

    let s1 = x - f32(i0);
    let s0 = 1.0 - s1;

    let t1 = y - f32(j0);
    let t0 = 1.0 - t1;

    let d00 = textureLoad(density_current, vec2<i32>(i0, j0));
    let d01 = textureLoad(density_current, vec2<i32>(i0, j1));
    let d10 = textureLoad(density_current, vec2<i32>(i1, j0));
    let d11 = textureLoad(density_current, vec2<i32>(i1, j1));

    var result = textureLoad(density_current, location);

    result.r =
        s0 * (t0 * d00.r + t1 * d01.r) +
        s1 * (t0 * d10.r + t1 * d11.r);

    result = set_bnd(location, result, 0);

    textureStore(density_next, location, result);
}

@compute @workgroup_size(8, 8, 1)
fn divergence(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
  // int N - The size
  // float * u - The x velocity component array
  // float * v - The y vleocity component array
  // float * p - The previous x velocity component (viscosity?)
  // float * div - The previous y velocity component?

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
    location.y == height - 1
  {
    // JAB TODO: Need to store some value in the texture here
      return;
  }

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
    location.y == height - 1
  {
      textureStore(
          density_next,
          location,
          textureLoad(density_current, location),
      );
      return;
  }
  let n = config.dimensions.x - 2.0;
  let a = config.time * config.diff * n * n;

  var x0 = textureLoad(density_original, location);

  let left = textureLoad(density_current, location + vec2<i32>(-1, 0));
  let right = textureLoad(density_current, location + vec2<i32>(1, 0));
  let below = textureLoad(density_current, location + vec2<i32>(0, -1));
  let above = textureLoad(density_current, location + vec2<i32>(0, 1));

  var result = textureLoad(density_current, location);

  result.r = (
      x0.r +
      a * (
          left.r +
          right.r +
          below.r +
          above.r
      )
  ) / (1.0 + 4.0 * a);

  textureStore(density_next, location, result);
}


@compute @workgroup_size(8, 8, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    if (invocation_id.x >= u32(config.dimensions.x) ||
        invocation_id.y >= u32(config.dimensions.y)) {
        return;
    }

    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    var color = textureLoad(density_current, location);
    color.r += config.time * density_source_at(location);
    color = set_bnd(location, color, 0);
    textureStore(density_next, location, color);
}
