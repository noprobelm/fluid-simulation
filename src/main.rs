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
};
use std::borrow::Cow;

const INIT_SHADER: &str = "shaders/init.wgsl";
const DENSITY_UPDATE_SHADER: &str = "shaders/density_update.wgsl";
const DENSITY_DIFFUSE_SHADER: &str = "shaders/density_diffuse.wgsl";
const DENSITY_ADVECT_SHADER: &str = "shaders/density_advect.wgsl";
const COMPUTE_DIVERGENCE_SHADER: &str = "shaders/divergence.wgsl";
const PRESSURE_SOLVE_SHADER: &str = "shaders/pressure_solve.wgsl";
const VELOCITY_PROJECT_SHADER: &str = "shaders/velocity_project.wgsl";

const DISPLAY_FACTOR: u32 = 4;
const SIZE: UVec2 = UVec2::new((1280 / DISPLAY_FACTOR) + 2, (1280 / DISPLAY_FACTOR) + 2);
const WORKGROUP_SIZE: u32 = 8;
const DIFFUSION_ITERATIONS: usize = 20;

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        resolution: (SIZE * DISPLAY_FACTOR).into(),
                        // present_mode: bevy::window::PresentMode::AutoNoVsync,
                        ..default()
                    }),
                    ..default()
                })
                .set(ImagePlugin::default_nearest()),
            FluidSimComputePlugin,
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, switch_textures)
        .add_systems(Update, (update_cursor_position, update_shader_time))
        .run();
}

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let mut density = Image::new_target_texture(SIZE.x, SIZE.y, TextureFormat::R32Float, None);
    density.asset_usage = RenderAssetUsages::RENDER_WORLD;
    density.texture_descriptor.usage = TextureUsages::COPY_SRC
        | TextureUsages::COPY_DST
        | TextureUsages::STORAGE_BINDING
        | TextureUsages::TEXTURE_BINDING;

    let density_original = images.add(density.clone());
    let density_current = images.add(density.clone());
    let density_next = images.add(density);

    let mut velocity = Image::new_target_texture(SIZE.x, SIZE.y, TextureFormat::Rg32Float, None);
    velocity.asset_usage = RenderAssetUsages::RENDER_WORLD;
    velocity.texture_descriptor.usage = TextureUsages::COPY_SRC
        | TextureUsages::COPY_DST
        | TextureUsages::STORAGE_BINDING
        | TextureUsages::TEXTURE_BINDING;

    let velocity_current = images.add(velocity.clone());
    let velocity_next = images.add(velocity);

    let mut divergence = Image::new_target_texture(SIZE.x, SIZE.y, TextureFormat::R32Float, None);
    divergence.asset_usage = RenderAssetUsages::RENDER_WORLD;
    divergence.texture_descriptor.usage = TextureUsages::STORAGE_BINDING;

    let divergence = images.add(divergence);

    let mut pressure = Image::new_target_texture(SIZE.x, SIZE.y, TextureFormat::R32Float, None);
    pressure.asset_usage = RenderAssetUsages::RENDER_WORLD;
    pressure.texture_descriptor.usage = TextureUsages::STORAGE_BINDING;

    let pressure_current = images.add(pressure.clone());
    let pressure_next = images.add(pressure);

    commands.insert_resource(FluidSimImages {
        density_original,
        density_current: density_current.clone(),
        density_next: density_next.clone(),
        velocity_current,
        velocity_next,
        divergence,
        pressure_current,
        pressure_next,
    });

    commands.insert_resource(FluidSimUniforms {
        alive_color: LinearRgba::RED,
        cursor_position: Vec2::ZERO,
        cursor_density: 10.0,
        time: 0.0,
        diff: 0.0001,
        dimensions: SIZE.as_vec2(),
    });

    // We'll temporarily display density until we have a better rendering solution
    commands.spawn((
        Sprite {
            image: density_current,
            custom_size: Some(SIZE.as_vec2()),
            ..default()
        },
        Transform::from_scale(Vec3::splat(DISPLAY_FACTOR as f32)),
    ));

    commands.spawn(Camera2d);
}

// We'll temporarily display density until we have a better rendering solution
fn switch_textures(images: Res<FluidSimImages>, mut sprite: Single<&mut Sprite>) {
    if sprite.image == images.density_current {
        sprite.image = images.density_next.clone();
    } else {
        sprite.image = images.density_current.clone();
    }
}

struct FluidSimComputePlugin;

impl Plugin for FluidSimComputePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            ExtractResourcePlugin::<FluidSimImages>::default(),
            ExtractResourcePlugin::<FluidSimUniforms>::default(),
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
    velocity: PingPong,
    pressure: PingPong,
}

#[derive(Resource, Clone, ExtractResource)]
struct FluidSimImages {
    density_original: Handle<Image>,
    density_current: Handle<Image>,
    density_next: Handle<Image>,
    velocity_current: Handle<Image>,
    velocity_next: Handle<Image>,
    divergence: Handle<Image>,
    pressure_current: Handle<Image>,
    pressure_next: Handle<Image>,
}

#[derive(Resource, Clone, ExtractResource, ShaderType)]
struct FluidSimUniforms {
    alive_color: LinearRgba,
    cursor_position: Vec2,
    cursor_density: f32,
    time: f32,
    diff: f32,
    dimensions: Vec2,
}

#[derive(Resource)]
struct FluidSimBindGroups {
    init: BindGroup,

    // [density ping-pong state]
    density_update: [BindGroup; 2],
    density_diffuse: [BindGroup; 2],

    // [density ping-pong state][velocity ping-pong state]
    density_advect: [[BindGroup; 2]; 2],

    compute_divergence: [BindGroup; 2],

    // [pressure ping-pong state]
    pressure_solve: [BindGroup; 2],

    // [velocity ping-pong state][pressure ping-pong state]
    velocity_project: [[BindGroup; 2]; 2],
}

fn prepare_bind_groups(
    mut commands: Commands,
    pipeline: Res<FluidSimPipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    fluid_sim_images: Res<FluidSimImages>,
    fluid_sim_uniforms: Res<FluidSimUniforms>,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    queue: Res<RenderQueue>,
) {
    let density_original = gpu_images.get(&fluid_sim_images.density_original).unwrap();
    let density_a = gpu_images.get(&fluid_sim_images.density_current).unwrap();
    let density_b = gpu_images.get(&fluid_sim_images.density_next).unwrap();

    let velocity_a = gpu_images.get(&fluid_sim_images.velocity_current).unwrap();
    let velocity_b = gpu_images.get(&fluid_sim_images.velocity_next).unwrap();

    let divergence = gpu_images.get(&fluid_sim_images.divergence).unwrap();

    let pressure_a = gpu_images.get(&fluid_sim_images.pressure_current).unwrap();
    let pressure_b = gpu_images.get(&fluid_sim_images.pressure_next).unwrap();

    let mut uniform_buffer = UniformBuffer::from(fluid_sim_uniforms.into_inner());
    uniform_buffer.write_buffer(&render_device, &queue);

    let init_layout = pipeline_cache.get_bind_group_layout(&pipeline.init_bind_group_layout);
    let density_update_layout =
        pipeline_cache.get_bind_group_layout(&pipeline.density_update_bind_group_layout);
    let density_diffuse_layout =
        pipeline_cache.get_bind_group_layout(&pipeline.density_diffuse_bind_group_layout);
    let density_advect_layout =
        pipeline_cache.get_bind_group_layout(&pipeline.density_advect_bind_group_layout);
    let divergence_layout =
        pipeline_cache.get_bind_group_layout(&pipeline.compute_divergence_bind_group_layout);
    let pressure_solve_layout =
        pipeline_cache.get_bind_group_layout(&pipeline.pressure_solve_bind_group_layout);
    let velocity_project_layout =
        pipeline_cache.get_bind_group_layout(&pipeline.velocity_project_bind_group_layout);

    // init.wgsl:
    //   0 density_a (write)
    //   1 density_b (write)
    //   2 velocity_a (write)
    //   3 velocity_b (write)
    //   4 config
    let init = render_device.create_bind_group(
        Some("fluid init bind group"),
        &init_layout,
        &BindGroupEntries::sequential((
            &density_a.texture_view,
            &density_b.texture_view,
            &velocity_a.texture_view,
            &velocity_b.texture_view,
            &uniform_buffer,
        )),
    );

    // density_update[0] = A -> B
    // density_update[1] = B -> A
    let density_update = [
        render_device.create_bind_group(
            Some("density update A -> B"),
            &density_update_layout,
            &BindGroupEntries::sequential((
                &density_a.texture_view,
                &density_b.texture_view,
                &uniform_buffer,
            )),
        ),
        render_device.create_bind_group(
            Some("density update B -> A"),
            &density_update_layout,
            &BindGroupEntries::sequential((
                &density_b.texture_view,
                &density_a.texture_view,
                &uniform_buffer,
            )),
        ),
    ];

    // density_diffuse[0] = source + A -> B
    // density_diffuse[1] = source + B -> A
    let density_diffuse = [
        render_device.create_bind_group(
            Some("density diffuse A -> B"),
            &density_diffuse_layout,
            &BindGroupEntries::sequential((
                &density_original.texture_view,
                &density_a.texture_view,
                &density_b.texture_view,
                &uniform_buffer,
            )),
        ),
        render_device.create_bind_group(
            Some("density diffuse B -> A"),
            &density_diffuse_layout,
            &BindGroupEntries::sequential((
                &density_original.texture_view,
                &density_b.texture_view,
                &density_a.texture_view,
                &uniform_buffer,
            )),
        ),
    ];

    // density_advect[density][velocity]
    let density_advect = [
        [
            // density A + velocity A -> density B
            render_device.create_bind_group(
                Some("density advect DA VA -> DB"),
                &density_advect_layout,
                &BindGroupEntries::sequential((
                    &density_a.texture_view,
                    &velocity_a.texture_view,
                    &density_b.texture_view,
                    &uniform_buffer,
                )),
            ),
            // density A + velocity B -> density B
            render_device.create_bind_group(
                Some("density advect DA VB -> DB"),
                &density_advect_layout,
                &BindGroupEntries::sequential((
                    &density_a.texture_view,
                    &velocity_b.texture_view,
                    &density_b.texture_view,
                    &uniform_buffer,
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
                    &velocity_a.texture_view,
                    &density_a.texture_view,
                    &uniform_buffer,
                )),
            ),
            // density B + velocity B -> density A
            render_device.create_bind_group(
                Some("density advect DB VB -> DA"),
                &density_advect_layout,
                &BindGroupEntries::sequential((
                    &density_b.texture_view,
                    &velocity_b.texture_view,
                    &density_a.texture_view,
                    &uniform_buffer,
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
                &uniform_buffer,
            )),
        ),
        render_device.create_bind_group(
            Some("compute divergence B"),
            &divergence_layout,
            &BindGroupEntries::sequential((
                &velocity_b.texture_view,
                &divergence.texture_view,
                &uniform_buffer,
            )),
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
                &uniform_buffer,
            )),
        ),
        render_device.create_bind_group(
            Some("pressure solve PB Di -> PA"),
            &pressure_solve_layout,
            &BindGroupEntries::sequential((
                &divergence.texture_view,
                &pressure_b.texture_view,
                &pressure_a.texture_view,
                &uniform_buffer,
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
                    &uniform_buffer,
                )),
            ),
            render_device.create_bind_group(
                Some("velocity project VA PB -> VB"),
                &velocity_project_layout,
                &BindGroupEntries::sequential((
                    &pressure_b.texture_view,
                    &velocity_a.texture_view,
                    &velocity_b.texture_view,
                    &uniform_buffer,
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
                    &uniform_buffer,
                )),
            ),
            render_device.create_bind_group(
                Some("velocity project VB PB -> VA"),
                &velocity_project_layout,
                &BindGroupEntries::sequential((
                    &pressure_b.texture_view,
                    &velocity_b.texture_view,
                    &velocity_a.texture_view,
                    &uniform_buffer,
                )),
            ),
        ],
    ];

    commands.insert_resource(FluidSimBindGroups {
        init,
        density_update,
        density_diffuse,
        density_advect,
        compute_divergence,
        pressure_solve,
        velocity_project,
    });
}

#[derive(Resource)]
struct FluidSimPipeline {
    init_bind_group_layout: BindGroupLayoutDescriptor,
    density_update_bind_group_layout: BindGroupLayoutDescriptor,
    density_diffuse_bind_group_layout: BindGroupLayoutDescriptor,
    density_advect_bind_group_layout: BindGroupLayoutDescriptor,
    compute_divergence_bind_group_layout: BindGroupLayoutDescriptor,
    pressure_solve_bind_group_layout: BindGroupLayoutDescriptor,
    velocity_project_bind_group_layout: BindGroupLayoutDescriptor,

    init_pipeline: CachedComputePipelineId,
    density_update_pipeline: CachedComputePipelineId,
    density_diffuse_pipeline: CachedComputePipelineId,
    density_advect_pipeline: CachedComputePipelineId,
    compute_divergence_pipeline: CachedComputePipelineId,
    pressure_solve_pipeline: CachedComputePipelineId,
    velocity_project_pipeline: CachedComputePipelineId,
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
                texture_storage_2d(TextureFormat::Rg32Float, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::Rg32Float, StorageTextureAccess::WriteOnly),
                uniform_buffer::<FluidSimUniforms>(false),
            ),
        ),
    );

    let density_update_bind_group_layout = BindGroupLayoutDescriptor::new(
        "density update layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                uniform_buffer::<FluidSimUniforms>(false),
            ),
        ),
    );

    let density_diffuse_bind_group_layout = BindGroupLayoutDescriptor::new(
        "density diffuse layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                uniform_buffer::<FluidSimUniforms>(false),
            ),
        ),
    );

    let density_advect_bind_group_layout = BindGroupLayoutDescriptor::new(
        "density advect layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::Rg32Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                uniform_buffer::<FluidSimUniforms>(false),
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
                uniform_buffer::<FluidSimUniforms>(false),
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
                uniform_buffer::<FluidSimUniforms>(false),
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
                uniform_buffer::<FluidSimUniforms>(false),
            ),
        ),
    );

    let init_shader = asset_server.load(INIT_SHADER);
    let density_update_shader = asset_server.load(DENSITY_UPDATE_SHADER);
    let density_diffuse_shader = asset_server.load(DENSITY_DIFFUSE_SHADER);
    let density_advect_shader = asset_server.load(DENSITY_ADVECT_SHADER);
    let compute_divergence_shader = asset_server.load(COMPUTE_DIVERGENCE_SHADER);
    let pressure_solve_shader = asset_server.load(PRESSURE_SOLVE_SHADER);
    let velocity_project_shader = asset_server.load(VELOCITY_PROJECT_SHADER);

    let init_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![init_bind_group_layout.clone()],
        shader: init_shader,
        entry_point: Some(Cow::Borrowed("main")),
        ..default()
    });

    let density_update_pipeline =
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            layout: vec![density_update_bind_group_layout.clone()],
            shader: density_update_shader,
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

    commands.insert_resource(FluidSimPipeline {
        init_bind_group_layout,
        density_update_bind_group_layout,
        density_diffuse_bind_group_layout,
        density_advect_bind_group_layout,
        compute_divergence_bind_group_layout,
        pressure_solve_bind_group_layout,
        velocity_project_bind_group_layout,
        init_pipeline,
        density_update_pipeline,
        density_diffuse_pipeline,
        density_advect_pipeline,
        compute_divergence_pipeline,
        pressure_solve_pipeline,
        velocity_project_pipeline,
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
            if pipeline_ready(&pipeline_cache, pipeline.density_update_pipeline)
                && pipeline_ready(&pipeline_cache, pipeline.density_diffuse_pipeline)
                && pipeline_ready(&pipeline_cache, pipeline.density_advect_pipeline)
                && pipeline_ready(&pipeline_cache, pipeline.compute_divergence_pipeline)
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
    mut buffers: ResMut<FluidSimBuffers>,
) {
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
                SIZE.x.div_ceil(WORKGROUP_SIZE),
                SIZE.y.div_ceil(WORKGROUP_SIZE),
                1,
            );
        }

        FluidSimState::Update => {
            // ------------------------------------------------------------
            // 1. Add density source: current density -> other density
            // ------------------------------------------------------------
            {
                let update_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.density_update_pipeline)
                    .unwrap();

                let density_index = buffers.density.index();

                let mut pass = render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor::default());

                pass.set_pipeline(update_pipeline);
                pass.set_bind_group(0, &bind_groups.density_update[density_index], &[]);
                pass.dispatch_workgroups(
                    SIZE.x.div_ceil(WORKGROUP_SIZE),
                    SIZE.y.div_ceil(WORKGROUP_SIZE),
                    1,
                );
            }
            buffers.density.swap();

            // ------------------------------------------------------------
            // 2. Copy the updated density into the fixed diffusion source x0
            // ------------------------------------------------------------
            let updated_density_handle = match buffers.density {
                PingPong::A => &fluid_sim_images.density_current,
                PingPong::B => &fluid_sim_images.density_next,
            };

            let updated_density = gpu_images.get(updated_density_handle).unwrap();
            let density_original = gpu_images.get(&fluid_sim_images.density_original).unwrap();

            render_context.command_encoder().copy_texture_to_texture(
                updated_density.texture.as_image_copy(),
                density_original.texture.as_image_copy(),
                Extent3d {
                    width: SIZE.x,
                    height: SIZE.y,
                    depth_or_array_layers: 1,
                },
            );

            // ------------------------------------------------------------
            // 3. Jacobi diffusion iterations
            // ------------------------------------------------------------
            {
                let diffuse_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.density_diffuse_pipeline)
                    .unwrap();

                let mut pass = render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor::default());

                pass.set_pipeline(diffuse_pipeline);

                for _ in 0..DIFFUSION_ITERATIONS {
                    let density_index = buffers.density.index();
                    pass.set_bind_group(0, &bind_groups.density_diffuse[density_index], &[]);
                    pass.dispatch_workgroups(
                        SIZE.x.div_ceil(WORKGROUP_SIZE),
                        SIZE.y.div_ceil(WORKGROUP_SIZE),
                        1,
                    );
                    buffers.density.swap();
                }
            }

            // ------------------------------------------------------------
            // 4. Advect density through the currently-valid velocity field
            // ------------------------------------------------------------
            {
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
                    SIZE.x.div_ceil(WORKGROUP_SIZE),
                    SIZE.y.div_ceil(WORKGROUP_SIZE),
                    1,
                );
            }
            buffers.density.swap();

            // ------------------------------------------------------------
            // 4. Comptue divergence
            // ------------------------------------------------------------

            {
                let velocity_index = buffers.velocity.index();

                let divergence_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.compute_divergence_pipeline)
                    .unwrap();

                let mut pass = render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor::default());

                pass.set_pipeline(divergence_pipeline);
                pass.set_bind_group(0, &bind_groups.compute_divergence[velocity_index], &[]);
                pass.dispatch_workgroups(
                    SIZE.x.div_ceil(WORKGROUP_SIZE),
                    SIZE.y.div_ceil(WORKGROUP_SIZE),
                    1,
                );
            }

            // ------------------------------------------------------------
            // 4. Solve for pressure
            // ------------------------------------------------------------
            {
                let pressure_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.pressure_solve_pipeline)
                    .unwrap();

                let pressure_index = buffers.pressure.index();

                let mut pass = render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor::default());

                pass.set_pipeline(pressure_pipeline);
                pass.set_bind_group(0, &bind_groups.pressure_solve[pressure_index], &[]);
                pass.dispatch_workgroups(
                    SIZE.x.div_ceil(WORKGROUP_SIZE),
                    SIZE.y.div_ceil(WORKGROUP_SIZE),
                    1,
                );
            }
            buffers.pressure.swap();

            // ------------------------------------------------------------
            // 4. Proejct velocity
            // ------------------------------------------------------------
            {
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
                    &bind_groups.velocity_project[pressure_index][velocity_index],
                    &[],
                );
                pass.dispatch_workgroups(
                    SIZE.x.div_ceil(WORKGROUP_SIZE),
                    SIZE.y.div_ceil(WORKGROUP_SIZE),
                    1,
                );
            }
            buffers.velocity.swap();
        }
    }
}

fn update_cursor_position(
    windows: Query<&Window>,
    mut uniforms: ResMut<FluidSimUniforms>,
) -> Result {
    let window = windows.single()?;

    if let Some(cursor_position) = window.cursor_position() {
        let texture_size = SIZE.as_vec2() * DISPLAY_FACTOR as f32;
        let texture_origin = (window.size() - texture_size) * 0.5;
        let relative = (cursor_position - texture_origin) / texture_size;

        uniforms.cursor_position = relative.clamp(Vec2::ZERO, Vec2::ONE);
    }

    Ok(())
}

fn update_shader_time(time: Res<Time>, mut uniforms: ResMut<FluidSimUniforms>) {
    uniforms.time = time.delta().as_secs_f32();
    uniforms.dimensions = SIZE.as_vec2();
}
