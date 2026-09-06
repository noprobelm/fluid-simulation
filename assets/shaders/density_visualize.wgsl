struct DensityVisualizeUniforms {
    color: vec4<f32>,
    fluid_intensity_scale: f32
}

@group(0) @binding(0)
var density: texture_storage_2d<r32float, read>;

@group(0) @binding(1)
var display: texture_storage_2d<rgba16float, write>;

@group(0) @binding(2)
var<uniform> config: DensityVisualizeUniforms;

fn smoke_color(density: f32) -> vec3<f32> {
    let intensity = 1.0 - exp(-max(density, 0.0) * config.fluid_intensity_scale);
    let smoke_tint = vec3<f32>(config.color.r, config.color.g, config.color.b);
    return smoke_tint * intensity;
}

@compute
@workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dimensions = textureDimensions(density);

    if (id.x >= dimensions.x || id.y >= dimensions.y) {
        return;
    }

    let p = vec2<i32>(id.xy);
    let value = max(textureLoad(density, p).r, 0.0);
    let color = smoke_color(value);

    textureStore(display, p, vec4<f32>(color, 1.0));
}
