mod resources;
mod signals;
mod uniforms;

use bevy::{
    asset::RenderAssetUsages,
    core_pipeline::schedule::camera_driver,
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_resource::{
            binding_types::{texture_storage_2d, uniform_buffer},
            *,
        },
        renderer::{RenderContext, RenderDevice, RenderGraph, RenderQueue},
        texture::GpuImage,
    },
    shader::ShaderCacheError,
    window::PrimaryWindow,
};
use std::borrow::Cow;

pub use crate::compute::uniforms::{AddSourcesUniforms, DensityVisualizeUniforms};

use self::uniforms::{
    AdvectionUniforms, DensityDiffuseUniforms, DimensionsUniforms, VelocityDiffuseUniforms,
};

pub use resources::*;
pub use signals::*;

const INIT_SHADER: &str = "shaders/init.wgsl";
const ADD_SOURCES_SHADER: &str = "shaders/add_sources.wgsl";
const DENSITY_DIFFUSE_SHADER: &str = "shaders/density_diffuse.wgsl";
const DENSITY_ADVECT_SHADER: &str = "shaders/density_advect.wgsl";
const VELOCITY_ADVECT_SHADER: &str = "shaders/velocity_advect.wgsl";
const COMPUTE_DIVERGENCE_SHADER: &str = "shaders/divergence.wgsl";
const DIVERGENCE_SET_BND_SHADER: &str = "shaders/divergence_set_bnd.wgsl";
const PRESSURE_CLEAR_SHADER: &str = "shaders/pressure_clear.wgsl";
const PRESSURE_SOLVE_SHADER: &str = "shaders/pressure_solve.wgsl";
const VELOCITY_DIFFUSE_SHADER: &str = "shaders/velocity_diffuse.wgsl";
const VELOCITY_PROJECT_SHADER: &str = "shaders/velocity_project.wgsl";
const DENSITY_VISUALIZE_SHADER: &str = "shaders/density_visualize.wgsl";

const WORKGROUP_SIZE: u32 = 8;

pub struct ComputePlugin;

impl Plugin for ComputePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DisplayFactor>()
            .init_resource::<Iterations>()
            .add_systems(Startup, setup)
            .add_plugins((
                signals::SignalsPlugin,
                uniforms::UniformsPlugin,
                ExtractResourcePlugin::<FluidSimImages>::default(),
                ExtractResourcePlugin::<Iterations>::default(),
            ));

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<FluidSimBuffers>()
            .init_resource::<FluidSimState>()
            .add_systems(RenderStartup, init_fluid_sim_pipeline)
            .add_systems(
                Render,
                prepare_bind_groups.in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(Render, update_pipeline_state.in_set(RenderSystems::Prepare))
            .add_systems(RenderGraph, fluid_simulation.before(camera_driver));
    }
}

#[derive(Copy, Clone, Default)]
enum PingPong {
    #[default]
    A,
    B,
}

impl PingPong {
    fn index(self) -> usize {
        match self {
            Self::A => 0,
            Self::B => 1,
        }
    }

    fn swap(&mut self) {
        *self = match self {
            Self::A => Self::B,
            Self::B => Self::A,
        };
    }
}

#[derive(Clone, Resource, Default)]
struct FluidSimBuffers {
    density: PingPong,
    dye: PingPong,
    velocity: PingPong,
    pressure: PingPong,
    reset_generation: u64,
}

#[derive(Resource, Clone, ExtractResource)]
struct FluidSimImages {
    density_original: Handle<Image>,
    density_current: Handle<Image>,
    density_next: Handle<Image>,
    dye_original: Handle<Image>,
    dye_current: Handle<Image>,
    dye_next: Handle<Image>,
    velocity_original: Handle<Image>,
    velocity_current: Handle<Image>,
    velocity_next: Handle<Image>,
    divergence: Handle<Image>,
    pressure_current: Handle<Image>,
    pressure_next: Handle<Image>,
    display: Handle<Image>,
}

#[derive(Resource)]
struct FluidSimBindGroups {
    init: BindGroup,

    // [density ping-pong state][velocity ping-pong state]
    add_sources: [[BindGroup; 2]; 2],
    density_diffuse: [BindGroup; 2],
    velocity_diffuse: [BindGroup; 2],
    velocity_advect: [BindGroup; 2],

    // [density ping-pong state][velocity ping-pong state]
    density_advect: [[BindGroup; 2]; 2],

    compute_divergence: [BindGroup; 2],
    divergence_set_bnd: [BindGroup; 2],

    // [pressure ping-pong state]
    pressure_clear: [BindGroup; 2],
    pressure_solve: [BindGroup; 2],

    // [velocity ping-pong state][pressure ping-pong state]
    velocity_project: [[BindGroup; 2]; 2],

    density_visualize: [BindGroup; 2],
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    display_factor: Res<DisplayFactor>,
) -> Result {
    let window = windows.single()?;
    let size = window.resolution.size().as_uvec2() / display_factor.0;

    let mut density = Image::new_target_texture(size.x, size.y, TextureFormat::R32Float, None);
    density.asset_usage = RenderAssetUsages::RENDER_WORLD;
    density.texture_descriptor.usage = TextureUsages::COPY_SRC
        | TextureUsages::COPY_DST
        | TextureUsages::STORAGE_BINDING
        | TextureUsages::TEXTURE_BINDING;

    let density_original = images.add(density.clone());
    let density_current = images.add(density.clone());
    let density_next = images.add(density);

    let mut dye = Image::new_target_texture(size.x, size.y, TextureFormat::Rgba16Float, None);
    dye.asset_usage = RenderAssetUsages::RENDER_WORLD;
    dye.texture_descriptor.usage =
        TextureUsages::COPY_SRC | TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING;

    let dye_original = images.add(dye.clone());
    let dye_current = images.add(dye.clone());
    let dye_next = images.add(dye);

    let mut velocity = Image::new_target_texture(size.x, size.y, TextureFormat::Rg32Float, None);
    velocity.asset_usage = RenderAssetUsages::RENDER_WORLD;
    velocity.texture_descriptor.usage = TextureUsages::COPY_SRC
        | TextureUsages::COPY_DST
        | TextureUsages::STORAGE_BINDING
        | TextureUsages::TEXTURE_BINDING;

    let velocity_original = images.add(velocity.clone());
    let velocity_current = images.add(velocity.clone());
    let velocity_next = images.add(velocity);

    let mut divergence = Image::new_target_texture(size.x, size.y, TextureFormat::R32Float, None);
    divergence.asset_usage = RenderAssetUsages::RENDER_WORLD;
    divergence.texture_descriptor.usage = TextureUsages::STORAGE_BINDING;

    let divergence = images.add(divergence);

    let mut pressure = Image::new_target_texture(size.x, size.y, TextureFormat::R32Float, None);
    pressure.asset_usage = RenderAssetUsages::RENDER_WORLD;
    pressure.texture_descriptor.usage = TextureUsages::STORAGE_BINDING;

    let pressure_current = images.add(pressure.clone());
    let pressure_next = images.add(pressure);

    let mut display = Image::new_target_texture(size.x, size.y, TextureFormat::Rgba16Float, None);
    display.asset_usage = RenderAssetUsages::RENDER_WORLD;
    display.texture_descriptor.usage =
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    let display = images.add(display);

    commands.insert_resource(FluidSimImages {
        density_original,
        density_current: density_current.clone(),
        density_next: density_next.clone(),
        dye_original,
        dye_current,
        dye_next,
        velocity_original,
        velocity_current,
        velocity_next,
        divergence,
        pressure_current,
        pressure_next,
        display: display.clone(),
    });

    commands.spawn((
        Sprite {
            image: display,
            custom_size: Some(size.as_vec2()),
            ..default()
        },
        Transform::from_scale(Vec3::splat(display_factor.0 as f32)),
    ));

    commands.spawn(Camera2d);

    Ok(())
}

fn prepare_bind_groups(
    mut commands: Commands,
    pipeline: Res<FluidSimPipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    fluid_sim_images: Res<FluidSimImages>,
    dimensions_uniforms: Res<DimensionsUniforms>,
    add_sources_uniforms: Res<AddSourcesUniforms>,
    advection_uniforms: Res<AdvectionUniforms>,
    density_diffuse_uniforms: Res<DensityDiffuseUniforms>,
    velocity_diffuse_uniforms: Res<VelocityDiffuseUniforms>,
    density_visualize_uniforms: Res<DensityVisualizeUniforms>,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    queue: Res<RenderQueue>,
) {
    let density_original = gpu_images.get(&fluid_sim_images.density_original).unwrap();
    let density_a = gpu_images.get(&fluid_sim_images.density_current).unwrap();
    let density_b = gpu_images.get(&fluid_sim_images.density_next).unwrap();

    let dye_a = gpu_images.get(&fluid_sim_images.dye_current).unwrap();
    let dye_b = gpu_images.get(&fluid_sim_images.dye_next).unwrap();
    let dye_original = gpu_images.get(&fluid_sim_images.dye_original).unwrap();

    let velocity_original = gpu_images.get(&fluid_sim_images.velocity_original).unwrap();
    let velocity_a = gpu_images.get(&fluid_sim_images.velocity_current).unwrap();
    let velocity_b = gpu_images.get(&fluid_sim_images.velocity_next).unwrap();

    let divergence = gpu_images.get(&fluid_sim_images.divergence).unwrap();

    let pressure_a = gpu_images.get(&fluid_sim_images.pressure_current).unwrap();
    let pressure_b = gpu_images.get(&fluid_sim_images.pressure_next).unwrap();
    let display = gpu_images.get(&fluid_sim_images.display).unwrap();

    let mut dimensions_uniform_buffer = UniformBuffer::from(dimensions_uniforms.into_inner());
    dimensions_uniform_buffer.write_buffer(&render_device, &queue);

    let mut add_sources_uniform_buffer = UniformBuffer::from(add_sources_uniforms.into_inner());
    add_sources_uniform_buffer.write_buffer(&render_device, &queue);

    let mut advection_uniform_buffer = UniformBuffer::from(advection_uniforms.into_inner());
    advection_uniform_buffer.write_buffer(&render_device, &queue);

    let mut density_diffuse_uniform_buffer =
        UniformBuffer::from(density_diffuse_uniforms.into_inner());
    density_diffuse_uniform_buffer.write_buffer(&render_device, &queue);

    let mut velocity_diffuse_uniform_buffer =
        UniformBuffer::from(velocity_diffuse_uniforms.into_inner());
    velocity_diffuse_uniform_buffer.write_buffer(&render_device, &queue);

    let mut density_visualize_uniform_buffer =
        UniformBuffer::from(density_visualize_uniforms.into_inner());
    density_visualize_uniform_buffer.write_buffer(&render_device, &queue);

    let init_layout = pipeline_cache.get_bind_group_layout(&pipeline.init_bind_group_layout);
    let add_sources_layout =
        pipeline_cache.get_bind_group_layout(&pipeline.add_sources_bind_group_layout);
    let density_diffuse_layout =
        pipeline_cache.get_bind_group_layout(&pipeline.density_diffuse_bind_group_layout);
    let velocity_diffuse_layout =
        pipeline_cache.get_bind_group_layout(&pipeline.velocity_diffuse_bind_group_layout);
    let velocity_advect_layout =
        pipeline_cache.get_bind_group_layout(&pipeline.velocity_advect_bind_group_layout);
    let density_advect_layout =
        pipeline_cache.get_bind_group_layout(&pipeline.density_advect_bind_group_layout);
    let divergence_layout =
        pipeline_cache.get_bind_group_layout(&pipeline.compute_divergence_bind_group_layout);
    let divergence_set_bnd_layout =
        pipeline_cache.get_bind_group_layout(&pipeline.divergence_set_bnd_bind_group_layout);
    let pressure_clear_layout =
        pipeline_cache.get_bind_group_layout(&pipeline.pressure_clear_bind_group_layout);
    let pressure_solve_layout =
        pipeline_cache.get_bind_group_layout(&pipeline.pressure_solve_bind_group_layout);
    let velocity_project_layout =
        pipeline_cache.get_bind_group_layout(&pipeline.velocity_project_bind_group_layout);
    let density_visualize_layout =
        pipeline_cache.get_bind_group_layout(&pipeline.density_visualize_bind_group_layout);

    // init.wgsl:
    //   0 density_a (write)
    //   1 density_b (write)
    //   2 dye_a (write)
    //   3 dye_b (write)
    //   4 velocity_a (write)
    //   5 velocity_b (write)
    //   6 display (write)
    //   7 config
    let init = render_device.create_bind_group(
        Some("fluid init bind group"),
        &init_layout,
        &BindGroupEntries::sequential((
            &density_a.texture_view,
            &density_b.texture_view,
            &dye_a.texture_view,
            &dye_b.texture_view,
            &velocity_a.texture_view,
            &velocity_b.texture_view,
            &display.texture_view,
            &dimensions_uniform_buffer,
        )),
    );

    // add_sources[density/dye][velocity]: each field's current texture -> its other texture
    let add_sources = [
        [
            render_device.create_bind_group(
                Some("add sources DA VA -> DB VB"),
                &add_sources_layout,
                &BindGroupEntries::sequential((
                    &density_a.texture_view,
                    &velocity_a.texture_view,
                    &dye_a.texture_view,
                    &density_b.texture_view,
                    &velocity_b.texture_view,
                    &dye_b.texture_view,
                    &add_sources_uniform_buffer,
                    &density_visualize_uniform_buffer,
                )),
            ),
            render_device.create_bind_group(
                Some("add sources DA VB -> DB VA"),
                &add_sources_layout,
                &BindGroupEntries::sequential((
                    &density_a.texture_view,
                    &velocity_b.texture_view,
                    &dye_a.texture_view,
                    &density_b.texture_view,
                    &velocity_a.texture_view,
                    &dye_b.texture_view,
                    &add_sources_uniform_buffer,
                    &density_visualize_uniform_buffer,
                )),
            ),
        ],
        [
            render_device.create_bind_group(
                Some("add sources DB VA -> DA VB"),
                &add_sources_layout,
                &BindGroupEntries::sequential((
                    &density_b.texture_view,
                    &velocity_a.texture_view,
                    &dye_b.texture_view,
                    &density_a.texture_view,
                    &velocity_b.texture_view,
                    &dye_a.texture_view,
                    &add_sources_uniform_buffer,
                    &density_visualize_uniform_buffer,
                )),
            ),
            render_device.create_bind_group(
                Some("add sources DB VB -> DA VA"),
                &add_sources_layout,
                &BindGroupEntries::sequential((
                    &density_b.texture_view,
                    &velocity_b.texture_view,
                    &dye_b.texture_view,
                    &density_a.texture_view,
                    &velocity_a.texture_view,
                    &dye_a.texture_view,
                    &add_sources_uniform_buffer,
                    &density_visualize_uniform_buffer,
                )),
            ),
        ],
    ];

    // density_diffuse[0] = density/dye source + A -> B
    // density_diffuse[1] = density/dye source + B -> A
    let density_diffuse = [
        render_device.create_bind_group(
            Some("density diffuse A -> B"),
            &density_diffuse_layout,
            &BindGroupEntries::sequential((
                &density_original.texture_view,
                &dye_original.texture_view,
                &density_a.texture_view,
                &dye_a.texture_view,
                &density_b.texture_view,
                &dye_b.texture_view,
                &density_diffuse_uniform_buffer,
            )),
        ),
        render_device.create_bind_group(
            Some("density diffuse B -> A"),
            &density_diffuse_layout,
            &BindGroupEntries::sequential((
                &density_original.texture_view,
                &dye_original.texture_view,
                &density_b.texture_view,
                &dye_b.texture_view,
                &density_a.texture_view,
                &dye_a.texture_view,
                &density_diffuse_uniform_buffer,
            )),
        ),
    ];

    // velocity_diffuse[0] = source + A -> B
    // velocity_diffuse[1] = source + B -> A
    let velocity_diffuse = [
        render_device.create_bind_group(
            Some("velocity diffuse A -> B"),
            &velocity_diffuse_layout,
            &BindGroupEntries::sequential((
                &velocity_original.texture_view,
                &velocity_a.texture_view,
                &velocity_b.texture_view,
                &velocity_diffuse_uniform_buffer,
            )),
        ),
        render_device.create_bind_group(
            Some("velocity diffuse B -> A"),
            &velocity_diffuse_layout,
            &BindGroupEntries::sequential((
                &velocity_original.texture_view,
                &velocity_b.texture_view,
                &velocity_a.texture_view,
                &velocity_diffuse_uniform_buffer,
            )),
        ),
    ];

    let velocity_advect = [
        render_device.create_bind_group(
            Some("velocity advect A -> B"),
            &velocity_advect_layout,
            &BindGroupEntries::sequential((
                &velocity_a.texture_view,
                &velocity_b.texture_view,
                &advection_uniform_buffer,
            )),
        ),
        render_device.create_bind_group(
            Some("velocity advect B -> A"),
            &velocity_advect_layout,
            &BindGroupEntries::sequential((
                &velocity_b.texture_view,
                &velocity_a.texture_view,
                &advection_uniform_buffer,
            )),
        ),
    ];

    // density_advect[density/dye][velocity]
    let density_advect = [
        [
            // density A + velocity A -> density B
            render_device.create_bind_group(
                Some("density advect DA VA -> DB"),
                &density_advect_layout,
                &BindGroupEntries::sequential((
                    &density_a.texture_view,
                    &dye_a.texture_view,
                    &velocity_a.texture_view,
                    &density_b.texture_view,
                    &dye_b.texture_view,
                    &advection_uniform_buffer,
                )),
            ),
            // density A + velocity B -> density B
            render_device.create_bind_group(
                Some("density advect DA VB -> DB"),
                &density_advect_layout,
                &BindGroupEntries::sequential((
                    &density_a.texture_view,
                    &dye_a.texture_view,
                    &velocity_b.texture_view,
                    &density_b.texture_view,
                    &dye_b.texture_view,
                    &advection_uniform_buffer,
                )),
            ),
        ],
        [
            // density B + velocity A -> density A
            render_device.create_bind_group(
                Some("density advect DB VA -> DA"),
                &density_advect_layout,
                &BindGroupEntries::sequential((
                    &density_b.texture_view,
                    &dye_b.texture_view,
                    &velocity_a.texture_view,
                    &density_a.texture_view,
                    &dye_a.texture_view,
                    &advection_uniform_buffer,
                )),
            ),
            // density B + velocity B -> density A
            render_device.create_bind_group(
                Some("density advect DB VB -> DA"),
                &density_advect_layout,
                &BindGroupEntries::sequential((
                    &density_b.texture_view,
                    &dye_b.texture_view,
                    &velocity_b.texture_view,
                    &density_a.texture_view,
                    &dye_a.texture_view,
                    &advection_uniform_buffer,
                )),
            ),
        ],
    ];

    let compute_divergence = [
        render_device.create_bind_group(
            Some("compute divergence A"),
            &divergence_layout,
            &BindGroupEntries::sequential((
                &velocity_a.texture_view,
                &divergence.texture_view,
                &dimensions_uniform_buffer,
            )),
        ),
        render_device.create_bind_group(
            Some("compute divergence B"),
            &divergence_layout,
            &BindGroupEntries::sequential((
                &velocity_b.texture_view,
                &divergence.texture_view,
                &dimensions_uniform_buffer,
            )),
        ),
    ];

    let divergence_set_bnd = [
        render_device.create_bind_group(
            Some("set divergence boundary A"),
            &divergence_set_bnd_layout,
            &BindGroupEntries::sequential((
                &velocity_a.texture_view,
                &divergence.texture_view,
                &dimensions_uniform_buffer,
            )),
        ),
        render_device.create_bind_group(
            Some("set divergence boundary B"),
            &divergence_set_bnd_layout,
            &BindGroupEntries::sequential((
                &velocity_b.texture_view,
                &divergence.texture_view,
                &dimensions_uniform_buffer,
            )),
        ),
    ];

    let pressure_clear = [
        render_device.create_bind_group(
            Some("clear pressure A"),
            &pressure_clear_layout,
            &BindGroupEntries::sequential((&pressure_a.texture_view, &dimensions_uniform_buffer)),
        ),
        render_device.create_bind_group(
            Some("clear pressure B"),
            &pressure_clear_layout,
            &BindGroupEntries::sequential((&pressure_b.texture_view, &dimensions_uniform_buffer)),
        ),
    ];

    let pressure_solve = [
        render_device.create_bind_group(
            Some("pressure solve PA Di -> PB"),
            &pressure_solve_layout,
            &BindGroupEntries::sequential((
                &divergence.texture_view,
                &pressure_a.texture_view,
                &pressure_b.texture_view,
                &dimensions_uniform_buffer,
            )),
        ),
        render_device.create_bind_group(
            Some("pressure solve PB Di -> PA"),
            &pressure_solve_layout,
            &BindGroupEntries::sequential((
                &divergence.texture_view,
                &pressure_b.texture_view,
                &pressure_a.texture_view,
                &dimensions_uniform_buffer,
            )),
        ),
    ];

    let velocity_project = [
        [
            render_device.create_bind_group(
                Some("velocity project VA PA -> VB"),
                &velocity_project_layout,
                &BindGroupEntries::sequential((
                    &pressure_a.texture_view,
                    &velocity_a.texture_view,
                    &velocity_b.texture_view,
                    &dimensions_uniform_buffer,
                )),
            ),
            render_device.create_bind_group(
                Some("velocity project VA PB -> VB"),
                &velocity_project_layout,
                &BindGroupEntries::sequential((
                    &pressure_b.texture_view,
                    &velocity_a.texture_view,
                    &velocity_b.texture_view,
                    &dimensions_uniform_buffer,
                )),
            ),
        ],
        [
            render_device.create_bind_group(
                Some("velocity project VB PA -> VA"),
                &velocity_project_layout,
                &BindGroupEntries::sequential((
                    &pressure_a.texture_view,
                    &velocity_b.texture_view,
                    &velocity_a.texture_view,
                    &dimensions_uniform_buffer,
                )),
            ),
            render_device.create_bind_group(
                Some("velocity project VB PB -> VA"),
                &velocity_project_layout,
                &BindGroupEntries::sequential((
                    &pressure_b.texture_view,
                    &velocity_b.texture_view,
                    &velocity_a.texture_view,
                    &dimensions_uniform_buffer,
                )),
            ),
        ],
    ];

    let density_visualize = [
        render_device.create_bind_group(
            Some("visualize density A"),
            &density_visualize_layout,
            &BindGroupEntries::sequential((
                &density_a.texture_view,
                &dye_a.texture_view,
                &display.texture_view,
                &density_visualize_uniform_buffer,
            )),
        ),
        render_device.create_bind_group(
            Some("visualize density B"),
            &density_visualize_layout,
            &BindGroupEntries::sequential((
                &density_b.texture_view,
                &dye_b.texture_view,
                &display.texture_view,
                &density_visualize_uniform_buffer,
            )),
        ),
    ];

    commands.insert_resource(FluidSimBindGroups {
        init,
        add_sources,
        density_diffuse,
        velocity_diffuse,
        velocity_advect,
        density_advect,
        compute_divergence,
        divergence_set_bnd,
        pressure_clear,
        pressure_solve,
        velocity_project,
        density_visualize,
    });
}

#[derive(Resource)]
struct FluidSimPipeline {
    init_bind_group_layout: BindGroupLayoutDescriptor,
    add_sources_bind_group_layout: BindGroupLayoutDescriptor,
    density_diffuse_bind_group_layout: BindGroupLayoutDescriptor,
    velocity_diffuse_bind_group_layout: BindGroupLayoutDescriptor,
    velocity_advect_bind_group_layout: BindGroupLayoutDescriptor,
    density_advect_bind_group_layout: BindGroupLayoutDescriptor,
    compute_divergence_bind_group_layout: BindGroupLayoutDescriptor,
    divergence_set_bnd_bind_group_layout: BindGroupLayoutDescriptor,
    pressure_clear_bind_group_layout: BindGroupLayoutDescriptor,
    pressure_solve_bind_group_layout: BindGroupLayoutDescriptor,
    velocity_project_bind_group_layout: BindGroupLayoutDescriptor,
    density_visualize_bind_group_layout: BindGroupLayoutDescriptor,

    init_pipeline: CachedComputePipelineId,
    add_sources_pipeline: CachedComputePipelineId,
    density_diffuse_pipeline: CachedComputePipelineId,
    velocity_diffuse_pipeline: CachedComputePipelineId,
    velocity_advect_pipeline: CachedComputePipelineId,
    density_advect_pipeline: CachedComputePipelineId,
    compute_divergence_pipeline: CachedComputePipelineId,
    divergence_set_bnd_pipeline: CachedComputePipelineId,
    pressure_clear_pipeline: CachedComputePipelineId,
    pressure_solve_pipeline: CachedComputePipelineId,
    velocity_project_pipeline: CachedComputePipelineId,
    density_visualize_pipeline: CachedComputePipelineId,
}

fn init_fluid_sim_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let init_bind_group_layout = BindGroupLayoutDescriptor::new(
        "fluid init layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::Rg32Float, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::Rg32Float, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                uniform_buffer::<DimensionsUniforms>(false),
            ),
        ),
    );

    let add_sources_bind_group_layout = BindGroupLayoutDescriptor::new(
        "add sources layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::Rg32Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::Rg32Float, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                uniform_buffer::<AddSourcesUniforms>(false),
                uniform_buffer::<DensityVisualizeUniforms>(false),
            ),
        ),
    );

    let density_diffuse_bind_group_layout = BindGroupLayoutDescriptor::new(
        "density diffuse layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                uniform_buffer::<DensityDiffuseUniforms>(false),
            ),
        ),
    );

    let velocity_diffuse_bind_group_layout = BindGroupLayoutDescriptor::new(
        "velocity diffuse layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::Rg32Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::Rg32Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::Rg32Float, StorageTextureAccess::WriteOnly),
                uniform_buffer::<VelocityDiffuseUniforms>(false),
            ),
        ),
    );

    let velocity_advect_bind_group_layout = BindGroupLayoutDescriptor::new(
        "velocity advect layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::Rg32Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::Rg32Float, StorageTextureAccess::WriteOnly),
                uniform_buffer::<AdvectionUniforms>(false),
            ),
        ),
    );

    let density_advect_bind_group_layout = BindGroupLayoutDescriptor::new(
        "density advect layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::Rg32Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                uniform_buffer::<AdvectionUniforms>(false),
            ),
        ),
    );

    let compute_divergence_bind_group_layout = BindGroupLayoutDescriptor::new(
        "compute divergence layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::Rg32Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                uniform_buffer::<DimensionsUniforms>(false),
            ),
        ),
    );

    let divergence_set_bnd_bind_group_layout = BindGroupLayoutDescriptor::new(
        "set divergence boundary layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::Rg32Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                uniform_buffer::<DimensionsUniforms>(false),
            ),
        ),
    );

    let pressure_clear_bind_group_layout = BindGroupLayoutDescriptor::new(
        "pressure clear layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                uniform_buffer::<DimensionsUniforms>(false),
            ),
        ),
    );

    let pressure_solve_bind_group_layout = BindGroupLayoutDescriptor::new(
        "pressure solve layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                uniform_buffer::<DimensionsUniforms>(false),
            ),
        ),
    );

    let velocity_project_bind_group_layout = BindGroupLayoutDescriptor::new(
        "velocity project layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::Rg32Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::Rg32Float, StorageTextureAccess::WriteOnly),
                uniform_buffer::<DimensionsUniforms>(false),
            ),
        ),
    );

    let density_visualize_bind_group_layout = BindGroupLayoutDescriptor::new(
        "density visualization layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                uniform_buffer::<DensityVisualizeUniforms>(false),
            ),
        ),
    );

    let init_shader = asset_server.load(INIT_SHADER);
    let add_sources_shader = asset_server.load(ADD_SOURCES_SHADER);
    let density_diffuse_shader = asset_server.load(DENSITY_DIFFUSE_SHADER);
    let velocity_diffuse_shader = asset_server.load(VELOCITY_DIFFUSE_SHADER);
    let velocity_advect_shader = asset_server.load(VELOCITY_ADVECT_SHADER);
    let density_advect_shader = asset_server.load(DENSITY_ADVECT_SHADER);
    let compute_divergence_shader = asset_server.load(COMPUTE_DIVERGENCE_SHADER);
    let divergence_set_bnd_shader = asset_server.load(DIVERGENCE_SET_BND_SHADER);
    let pressure_clear_shader = asset_server.load(PRESSURE_CLEAR_SHADER);
    let pressure_solve_shader = asset_server.load(PRESSURE_SOLVE_SHADER);
    let velocity_project_shader = asset_server.load(VELOCITY_PROJECT_SHADER);
    let density_visualize_shader = asset_server.load(DENSITY_VISUALIZE_SHADER);

    let init_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![init_bind_group_layout.clone()],
        shader: init_shader,
        entry_point: Some(Cow::Borrowed("main")),
        ..default()
    });

    let add_sources_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![add_sources_bind_group_layout.clone()],
        shader: add_sources_shader,
        entry_point: Some(Cow::Borrowed("main")),
        ..default()
    });

    let density_diffuse_pipeline =
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            layout: vec![density_diffuse_bind_group_layout.clone()],
            shader: density_diffuse_shader,
            entry_point: Some(Cow::Borrowed("main")),
            ..default()
        });

    let velocity_diffuse_pipeline =
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            layout: vec![velocity_diffuse_bind_group_layout.clone()],
            shader: velocity_diffuse_shader,
            entry_point: Some(Cow::Borrowed("main")),
            ..default()
        });

    let velocity_advect_pipeline =
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            layout: vec![velocity_advect_bind_group_layout.clone()],
            shader: velocity_advect_shader,
            entry_point: Some(Cow::Borrowed("main")),
            ..default()
        });

    let density_advect_pipeline =
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            layout: vec![density_advect_bind_group_layout.clone()],
            shader: density_advect_shader,
            entry_point: Some(Cow::Borrowed("main")),
            ..default()
        });

    let compute_divergence_pipeline =
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            layout: vec![compute_divergence_bind_group_layout.clone()],
            shader: compute_divergence_shader,
            entry_point: Some(Cow::Borrowed("main")),
            ..default()
        });

    let divergence_set_bnd_pipeline =
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            layout: vec![divergence_set_bnd_bind_group_layout.clone()],
            shader: divergence_set_bnd_shader,
            entry_point: Some(Cow::Borrowed("main")),
            ..default()
        });

    let pressure_clear_pipeline =
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            layout: vec![pressure_clear_bind_group_layout.clone()],
            shader: pressure_clear_shader,
            entry_point: Some(Cow::Borrowed("main")),
            ..default()
        });

    let pressure_solve_pipeline =
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            layout: vec![pressure_solve_bind_group_layout.clone()],
            shader: pressure_solve_shader,
            entry_point: Some(Cow::Borrowed("main")),
            ..default()
        });

    let velocity_project_pipeline =
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            layout: vec![velocity_project_bind_group_layout.clone()],
            shader: velocity_project_shader,
            entry_point: Some(Cow::Borrowed("main")),
            ..default()
        });

    let density_visualize_pipeline =
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            layout: vec![density_visualize_bind_group_layout.clone()],
            shader: density_visualize_shader,
            entry_point: Some(Cow::Borrowed("main")),
            ..default()
        });

    commands.insert_resource(FluidSimPipeline {
        init_bind_group_layout,
        add_sources_bind_group_layout,
        density_diffuse_bind_group_layout,
        velocity_diffuse_bind_group_layout,
        velocity_advect_bind_group_layout,
        density_advect_bind_group_layout,
        compute_divergence_bind_group_layout,
        divergence_set_bnd_bind_group_layout,
        pressure_clear_bind_group_layout,
        pressure_solve_bind_group_layout,
        velocity_project_bind_group_layout,
        density_visualize_bind_group_layout,
        init_pipeline,
        add_sources_pipeline,
        density_diffuse_pipeline,
        velocity_diffuse_pipeline,
        velocity_advect_pipeline,
        density_advect_pipeline,
        compute_divergence_pipeline,
        divergence_set_bnd_pipeline,
        pressure_clear_pipeline,
        pressure_solve_pipeline,
        velocity_project_pipeline,
        density_visualize_pipeline,
    });
}

#[derive(Resource, Default)]
enum FluidSimState {
    #[default]
    Loading,
    Init,
    Update,
}

fn pipeline_ready(pipeline_cache: &PipelineCache, id: CachedComputePipelineId) -> bool {
    matches!(
        pipeline_cache.get_compute_pipeline_state(id),
        CachedPipelineState::Ok(_)
    )
}

fn update_pipeline_state(
    pipeline: Res<FluidSimPipeline>,
    pipeline_cache: Res<PipelineCache>,
    mut state: ResMut<FluidSimState>,
) {
    match *state {
        FluidSimState::Loading => {
            match pipeline_cache.get_compute_pipeline_state(pipeline.init_pipeline) {
                CachedPipelineState::Ok(_) => {
                    *state = FluidSimState::Init;
                }
                CachedPipelineState::Err(ShaderCacheError::ShaderNotLoaded(_)) => {}
                CachedPipelineState::Err(err) => {
                    panic!("Initializing {INIT_SHADER}:\n{err}")
                }
                _ => {}
            }
        }
        FluidSimState::Init => {
            if pipeline_ready(&pipeline_cache, pipeline.add_sources_pipeline)
                && pipeline_ready(&pipeline_cache, pipeline.density_diffuse_pipeline)
                && pipeline_ready(&pipeline_cache, pipeline.velocity_diffuse_pipeline)
                && pipeline_ready(&pipeline_cache, pipeline.velocity_advect_pipeline)
                && pipeline_ready(&pipeline_cache, pipeline.density_advect_pipeline)
                && pipeline_ready(&pipeline_cache, pipeline.compute_divergence_pipeline)
                && pipeline_ready(&pipeline_cache, pipeline.divergence_set_bnd_pipeline)
                && pipeline_ready(&pipeline_cache, pipeline.pressure_clear_pipeline)
                && pipeline_ready(&pipeline_cache, pipeline.pressure_solve_pipeline)
                && pipeline_ready(&pipeline_cache, pipeline.velocity_project_pipeline)
                && pipeline_ready(&pipeline_cache, pipeline.density_visualize_pipeline)
            {
                *state = FluidSimState::Update;
            }
        }
        FluidSimState::Update => {}
    }
}

fn fluid_simulation(
    mut render_context: RenderContext,
    bind_groups: Res<FluidSimBindGroups>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<FluidSimPipeline>,
    state: Res<FluidSimState>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    fluid_sim_images: Res<FluidSimImages>,
    uniforms: Res<DimensionsUniforms>,
    mut buffers: ResMut<FluidSimBuffers>,
    reset: Res<ResetSimulation>,
    iterations: Res<Iterations>,
) {
    let size = uniforms.dimensions.as_uvec2();

    match *state {
        FluidSimState::Loading => {}

        FluidSimState::Init => {
            let init_pipeline = pipeline_cache
                .get_compute_pipeline(pipeline.init_pipeline)
                .unwrap();

            let mut pass = render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor::default());

            pass.set_pipeline(init_pipeline);
            pass.set_bind_group(0, &bind_groups.init, &[]);
            pass.dispatch_workgroups(
                size.x.div_ceil(WORKGROUP_SIZE),
                size.y.div_ceil(WORKGROUP_SIZE),
                1,
            );
            buffers.reset_generation = reset.generation;
        }

        FluidSimState::Update => {
            if buffers.reset_generation != reset.generation {
                let init_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.init_pipeline)
                    .unwrap();
                let mut pass = render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor::default());

                pass.set_pipeline(init_pipeline);
                pass.set_bind_group(0, &bind_groups.init, &[]);
                pass.dispatch_workgroups(
                    size.x.div_ceil(WORKGROUP_SIZE),
                    size.y.div_ceil(WORKGROUP_SIZE),
                    1,
                );

                *buffers = FluidSimBuffers {
                    reset_generation: reset.generation,
                    ..default()
                };
                return;
            }

            add_sources(
                size,
                &pipeline_cache,
                &mut buffers,
                &mut render_context,
                &pipeline,
                &bind_groups,
            );
            copy_current_velocity(
                size,
                &gpu_images,
                &fluid_sim_images,
                &buffers,
                &mut render_context,
            );
            diffuse_velocity(
                size,
                &pipeline_cache,
                &mut buffers,
                &mut render_context,
                &pipeline,
                &bind_groups,
                iterations.velocity_diffusion,
            );
            project_velocity(
                size,
                &pipeline_cache,
                &mut buffers,
                &mut render_context,
                &pipeline,
                &bind_groups,
                iterations.pressure_solve,
            );
            advect_velocity(
                size,
                &pipeline_cache,
                &mut buffers,
                &mut render_context,
                &pipeline,
                &bind_groups,
            );
            project_velocity(
                size,
                &pipeline_cache,
                &mut buffers,
                &mut render_context,
                &pipeline,
                &bind_groups,
                iterations.pressure_solve,
            );
            copy_current_density(
                size,
                &gpu_images,
                &fluid_sim_images,
                &buffers,
                &mut render_context,
            );
            diffuse_density(
                size,
                &pipeline_cache,
                &mut buffers,
                &mut render_context,
                &pipeline,
                &bind_groups,
                iterations.density_diffusion,
            );
            advect_density(
                size,
                &pipeline_cache,
                &mut buffers,
                &mut render_context,
                &pipeline,
                &bind_groups,
            );
            visualize_density(
                size,
                &pipeline_cache,
                &buffers,
                &mut render_context,
                &pipeline,
                &bind_groups,
            );
        }
    }
}

fn add_sources(
    size: UVec2,
    pipeline_cache: &Res<PipelineCache>,
    buffers: &mut ResMut<FluidSimBuffers>,
    render_context: &mut RenderContext,
    pipeline: &Res<FluidSimPipeline>,
    bind_groups: &Res<FluidSimBindGroups>,
) {
    let add_sources_pipeline = pipeline_cache
        .get_compute_pipeline(pipeline.add_sources_pipeline)
        .unwrap();

    let density_index = buffers.density.index();
    let velocity_index = buffers.velocity.index();

    let mut pass = render_context
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor::default());

    pass.set_pipeline(add_sources_pipeline);
    pass.set_bind_group(
        0,
        &bind_groups.add_sources[density_index][velocity_index],
        &[],
    );
    pass.dispatch_workgroups(
        size.x.div_ceil(WORKGROUP_SIZE),
        size.y.div_ceil(WORKGROUP_SIZE),
        1,
    );
    buffers.density.swap();
    buffers.dye.swap();
    buffers.velocity.swap();
}

fn copy_current_velocity(
    size: UVec2,
    gpu_images: &Res<RenderAssets<GpuImage>>,
    images: &Res<FluidSimImages>,
    buffers: &ResMut<FluidSimBuffers>,
    render_context: &mut RenderContext,
) {
    let current_handle = match buffers.velocity {
        PingPong::A => &images.velocity_current,
        PingPong::B => &images.velocity_next,
    };
    let current = gpu_images.get(current_handle).unwrap();
    let original = gpu_images.get(&images.velocity_original).unwrap();

    render_context.command_encoder().copy_texture_to_texture(
        current.texture.as_image_copy(),
        original.texture.as_image_copy(),
        Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
    );
}

fn copy_current_density(
    size: UVec2,
    gpu_images: &Res<RenderAssets<GpuImage>>,
    images: &Res<FluidSimImages>,
    buffers: &ResMut<FluidSimBuffers>,
    render_context: &mut RenderContext,
) {
    let current_handle = match buffers.density {
        PingPong::A => &images.density_current,
        PingPong::B => &images.density_next,
    };
    let current = gpu_images.get(current_handle).unwrap();
    let original = gpu_images.get(&images.density_original).unwrap();

    let current_dye_handle = match buffers.dye {
        PingPong::A => &images.dye_current,
        PingPong::B => &images.dye_next,
    };
    let current_dye = gpu_images.get(current_dye_handle).unwrap();
    let original_dye = gpu_images.get(&images.dye_original).unwrap();

    render_context.command_encoder().copy_texture_to_texture(
        current.texture.as_image_copy(),
        original.texture.as_image_copy(),
        Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
    );
    render_context.command_encoder().copy_texture_to_texture(
        current_dye.texture.as_image_copy(),
        original_dye.texture.as_image_copy(),
        Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
    );
}

fn diffuse_density(
    size: UVec2,
    pipeline_cache: &Res<PipelineCache>,
    buffers: &mut ResMut<FluidSimBuffers>,
    render_context: &mut RenderContext,
    pipeline: &Res<FluidSimPipeline>,
    bind_groups: &Res<FluidSimBindGroups>,
    iterations: usize,
) {
    let diffuse_pipeline = pipeline_cache
        .get_compute_pipeline(pipeline.density_diffuse_pipeline)
        .unwrap();

    let mut pass = render_context
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor::default());

    pass.set_pipeline(diffuse_pipeline);

    for _ in 0..iterations {
        let density_index = buffers.density.index();
        pass.set_bind_group(0, &bind_groups.density_diffuse[density_index], &[]);
        pass.dispatch_workgroups(
            size.x.div_ceil(WORKGROUP_SIZE),
            size.y.div_ceil(WORKGROUP_SIZE),
            1,
        );
        buffers.density.swap();
        buffers.dye.swap();
    }
}

fn diffuse_velocity(
    size: UVec2,
    pipeline_cache: &Res<PipelineCache>,
    buffers: &mut ResMut<FluidSimBuffers>,
    render_context: &mut RenderContext,
    pipeline: &Res<FluidSimPipeline>,
    bind_groups: &Res<FluidSimBindGroups>,
    iterations: usize,
) {
    let diffuse_pipeline = pipeline_cache
        .get_compute_pipeline(pipeline.velocity_diffuse_pipeline)
        .unwrap();

    let mut pass = render_context
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor::default());

    pass.set_pipeline(diffuse_pipeline);

    for _ in 0..iterations {
        let velocity_index = buffers.velocity.index();
        pass.set_bind_group(0, &bind_groups.velocity_diffuse[velocity_index], &[]);
        pass.dispatch_workgroups(
            size.x.div_ceil(WORKGROUP_SIZE),
            size.y.div_ceil(WORKGROUP_SIZE),
            1,
        );
        buffers.velocity.swap();
    }
}

fn advect_velocity(
    size: UVec2,
    pipeline_cache: &Res<PipelineCache>,
    buffers: &mut ResMut<FluidSimBuffers>,
    render_context: &mut RenderContext,
    pipeline: &Res<FluidSimPipeline>,
    bind_groups: &Res<FluidSimBindGroups>,
) {
    let advect_pipeline = pipeline_cache
        .get_compute_pipeline(pipeline.velocity_advect_pipeline)
        .unwrap();
    let velocity_index = buffers.velocity.index();

    let mut pass = render_context
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor::default());
    pass.set_pipeline(advect_pipeline);
    pass.set_bind_group(0, &bind_groups.velocity_advect[velocity_index], &[]);
    pass.dispatch_workgroups(
        size.x.div_ceil(WORKGROUP_SIZE),
        size.y.div_ceil(WORKGROUP_SIZE),
        1,
    );
    buffers.velocity.swap();
}

fn advect_density(
    size: UVec2,
    pipeline_cache: &Res<PipelineCache>,
    buffers: &mut ResMut<FluidSimBuffers>,
    render_context: &mut RenderContext,
    pipeline: &Res<FluidSimPipeline>,
    bind_groups: &Res<FluidSimBindGroups>,
) {
    let advect_pipeline = pipeline_cache
        .get_compute_pipeline(pipeline.density_advect_pipeline)
        .unwrap();
    let density_index = buffers.density.index();
    let velocity_index = buffers.velocity.index();

    let mut pass = render_context
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor::default());
    pass.set_pipeline(advect_pipeline);
    pass.set_bind_group(
        0,
        &bind_groups.density_advect[density_index][velocity_index],
        &[],
    );
    pass.dispatch_workgroups(
        size.x.div_ceil(WORKGROUP_SIZE),
        size.y.div_ceil(WORKGROUP_SIZE),
        1,
    );
    buffers.density.swap();
    buffers.dye.swap();
}

fn visualize_density(
    size: UVec2,
    pipeline_cache: &Res<PipelineCache>,
    buffers: &ResMut<FluidSimBuffers>,
    render_context: &mut RenderContext,
    pipeline: &Res<FluidSimPipeline>,
    bind_groups: &Res<FluidSimBindGroups>,
) {
    let visualize_pipeline = pipeline_cache
        .get_compute_pipeline(pipeline.density_visualize_pipeline)
        .unwrap();
    let density_index = buffers.density.index();
    let dye_index = buffers.dye.index();

    debug_assert_eq!(density_index, dye_index);

    let mut pass = render_context
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor::default());
    pass.set_pipeline(visualize_pipeline);
    pass.set_bind_group(0, &bind_groups.density_visualize[density_index], &[]);
    pass.dispatch_workgroups(
        size.x.div_ceil(WORKGROUP_SIZE),
        size.y.div_ceil(WORKGROUP_SIZE),
        1,
    );
}

fn project_velocity(
    size: UVec2,
    pipeline_cache: &Res<PipelineCache>,
    buffers: &mut ResMut<FluidSimBuffers>,
    render_context: &mut RenderContext,
    pipeline: &Res<FluidSimPipeline>,
    bind_groups: &Res<FluidSimBindGroups>,
    iterations: usize,
) {
    let velocity_index = buffers.velocity.index();
    let divergence_pipeline = pipeline_cache
        .get_compute_pipeline(pipeline.compute_divergence_pipeline)
        .unwrap();
    {
        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_pipeline(divergence_pipeline);
        pass.set_bind_group(0, &bind_groups.compute_divergence[velocity_index], &[]);
        pass.dispatch_workgroups(
            size.x.div_ceil(WORKGROUP_SIZE),
            size.y.div_ceil(WORKGROUP_SIZE),
            1,
        );
    }

    let divergence_set_bnd_pipeline = pipeline_cache
        .get_compute_pipeline(pipeline.divergence_set_bnd_pipeline)
        .unwrap();
    {
        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_pipeline(divergence_set_bnd_pipeline);
        pass.set_bind_group(0, &bind_groups.divergence_set_bnd[velocity_index], &[]);
        pass.dispatch_workgroups(
            size.x.div_ceil(WORKGROUP_SIZE),
            size.y.div_ceil(WORKGROUP_SIZE),
            1,
        );
    }

    let pressure_clear_pipeline = pipeline_cache
        .get_compute_pipeline(pipeline.pressure_clear_pipeline)
        .unwrap();
    {
        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_pipeline(pressure_clear_pipeline);
        for pressure_clear_bind_group in &bind_groups.pressure_clear {
            pass.set_bind_group(0, pressure_clear_bind_group, &[]);
            pass.dispatch_workgroups(
                size.x.div_ceil(WORKGROUP_SIZE),
                size.y.div_ceil(WORKGROUP_SIZE),
                1,
            );
        }
    }

    let pressure_pipeline = pipeline_cache
        .get_compute_pipeline(pipeline.pressure_solve_pipeline)
        .unwrap();
    {
        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_pipeline(pressure_pipeline);
        for _ in 0..iterations {
            let pressure_index = buffers.pressure.index();
            pass.set_bind_group(0, &bind_groups.pressure_solve[pressure_index], &[]);
            pass.dispatch_workgroups(
                size.x.div_ceil(WORKGROUP_SIZE),
                size.y.div_ceil(WORKGROUP_SIZE),
                1,
            );
            buffers.pressure.swap();
        }
    }

    let velocity_pipeline = pipeline_cache
        .get_compute_pipeline(pipeline.velocity_project_pipeline)
        .unwrap();

    let pressure_index = buffers.pressure.index();
    let velocity_index = buffers.velocity.index();

    let mut pass = render_context
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor::default());

    pass.set_pipeline(velocity_pipeline);
    pass.set_bind_group(
        0,
        &bind_groups.velocity_project[velocity_index][pressure_index],
        &[],
    );
    pass.dispatch_workgroups(
        size.x.div_ceil(WORKGROUP_SIZE),
        size.y.div_ceil(WORKGROUP_SIZE),
        1,
    );
    buffers.velocity.swap();
}
