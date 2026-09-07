struct DimensionsUniforms {
    dimensions: vec2<f32>,
};


@group(0) @binding(0)
var density_a: texture_storage_2d<r32float, write>;

@group(0) @binding(1)
var density_b: texture_storage_2d<r32float, write>;

@group(0) @binding(2)
var dye_a: texture_storage_2d<rgba16float, write>;

@group(0) @binding(3)
var dye_b: texture_storage_2d<rgba16float, write>;

@group(0) @binding(4)
var<uniform> config: DimensionsUniforms;

@compute
@workgroup_size(8, 8, 1)
fn init_density_dye(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= u32(config.dimensions.x) || id.y >= u32(config.dimensions.y)) {
        return;
    }

    let p = vec2<i32>(id.xy);

    textureStore(density_a, p, vec4<f32>(0.0));
    textureStore(density_b, p, vec4<f32>(0.0));
    textureStore(dye_a, p, vec4<f32>(0.0));
    textureStore(dye_b, p, vec4<f32>(0.0));
}
