struct DensityVisualizeUniforms {
    color: vec4<f32>,
    fluid_intensity_scale: f32
}

@group(0) @binding(0)
var density: texture_storage_2d<r32float, read>;

@group(0) @binding(1)
var dye: texture_storage_2d<rgba16float, read>;

@group(0) @binding(2)
var display: texture_storage_2d<rgba16float, write>;

@group(0) @binding(3)
var<uniform> config: DensityVisualizeUniforms;

fn fluid_color(p: vec2<i32>) -> vec3<f32> {
    let value = max(textureLoad(density, p).r, 0.0);
    let intensity = 1.0 - exp(-max(value, 0.0) * config.fluid_intensity_scale);
    let pigment = textureLoad(dye, p).rgb;
    let tint = pigment / max(value, 0.0001);
    return tint * intensity;
}

@compute
@workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dimensions = textureDimensions(density);

    if (id.x >= dimensions.x || id.y >= dimensions.y) {
        return;
    }

    let p = vec2<i32>(id.xy);
    let color = fluid_color(p);

    textureStore(display, p, vec4<f32>(color, 1.0));
}
