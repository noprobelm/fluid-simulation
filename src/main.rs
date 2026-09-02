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

/// This example uses a shader source file from the assets subdirectory
const SHADER_ASSET_PATH: &str = "shaders/fluid_simulation.wgsl";

const DISPLAY_FACTOR: u32 = 4;
const SIZE: UVec2 = UVec2::new((1280 / DISPLAY_FACTOR) + 2, (1280 / DISPLAY_FACTOR) + 2);
const WORKGROUP_SIZE: u32 = 8;

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        resolution: (SIZE * DISPLAY_FACTOR).into(),
                        // uncomment for unthrottled FPS
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
    let mut image = Image::new_target_texture(SIZE.x, SIZE.y, TextureFormat::Rgba32Float, None);
    image.asset_usage = RenderAssetUsages::RENDER_WORLD;
    image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    let original = images.add(image.clone());
    let current = images.add(image.clone());
    let next = images.add(image);

    commands.spawn((
        Sprite {
            image: next.clone(),
            custom_size: Some(SIZE.as_vec2()),
            ..default()
        },
        Transform::from_scale(Vec3::splat(DISPLAY_FACTOR as f32)),
    ));
    commands.spawn(Camera2d);

    commands.insert_resource(FluidSimImages {
        original,
        current,
        next,
    });

    commands.insert_resource(FluidSimUniforms {
        alive_color: LinearRgba::RED,
        cursor_position: Vec2::ZERO,
        cursor_density: 10.,
        time: 0.0,
        diff: 0.0001,
        dimensions: SIZE.as_vec2(),
    });
}

// Switch texture to display every frame to show the one that was written to most recently.
fn switch_textures(images: Res<FluidSimImages>, mut sprite: Single<&mut Sprite>) {
    if sprite.image == images.current {
        sprite.image = images.next.clone();
    } else {
        sprite.image = images.current.clone();
    }
}

struct FluidSimComputePlugin;

impl Plugin for FluidSimComputePlugin {
    fn build(&self, app: &mut App) {
        // Extract the game of life image resource from the main world into the render world
        // for operation on by the compute shader and display on the sprite.
        app.add_plugins((
            ExtractResourcePlugin::<FluidSimImages>::default(),
            ExtractResourcePlugin::<FluidSimUniforms>::default(),
        ));
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<FluidSimState>()
            .add_systems(RenderStartup, init_fluid_sim_pipeline)
            .add_systems(
                Render,
                prepare_bind_group.in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(Render, update.in_set(RenderSystems::Prepare))
            .add_systems(RenderGraph, fluid_simulation.before(camera_driver));
    }
}

#[derive(Resource, Clone, ExtractResource)]
struct FluidSimImages {
    original: Handle<Image>,
    current: Handle<Image>,
    next: Handle<Image>,
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
struct FluidSimBindGroups([BindGroup; 2]);

fn prepare_bind_group(
    mut commands: Commands,
    pipeline: Res<FluidSimPipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    fluid_sim_images: Res<FluidSimImages>,
    fluid_sim_uniforms: Res<FluidSimUniforms>,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    queue: Res<RenderQueue>,
) {
    let view_a = gpu_images.get(&fluid_sim_images.original).unwrap();
    let view_b = gpu_images.get(&fluid_sim_images.current).unwrap();
    let view_c = gpu_images.get(&fluid_sim_images.next).unwrap();

    let mut uniform_buffer = UniformBuffer::from(fluid_sim_uniforms.into_inner());
    uniform_buffer.write_buffer(&render_device, &queue);

    let bind_group_0 = render_device.create_bind_group(
        None,
        &pipeline_cache.get_bind_group_layout(&pipeline.texture_bind_group_layout),
        &BindGroupEntries::sequential((
            &view_a.texture_view,
            &view_b.texture_view,
            &view_c.texture_view,
            &uniform_buffer,
        )),
    );
    let bind_group_1 = render_device.create_bind_group(
        None,
        &pipeline_cache.get_bind_group_layout(&pipeline.texture_bind_group_layout),
        &BindGroupEntries::sequential((
            &view_a.texture_view,
            &view_c.texture_view,
            &view_b.texture_view,
            &uniform_buffer,
        )),
    );
    commands.insert_resource(FluidSimBindGroups([bind_group_0, bind_group_1]));
}

#[derive(Resource)]
struct FluidSimPipeline {
    texture_bind_group_layout: BindGroupLayoutDescriptor,
    init_pipeline: CachedComputePipelineId,
    update_pipeline: CachedComputePipelineId,
    diffuse_pipeline: CachedComputePipelineId,
}

fn init_fluid_sim_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let texture_bind_group_layout = BindGroupLayoutDescriptor::new(
        "FluidSimImages",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                uniform_buffer::<FluidSimUniforms>(false),
            ),
        ),
    );
    let shader = asset_server.load(SHADER_ASSET_PATH);
    let init_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![texture_bind_group_layout.clone()],
        shader: shader.clone(),
        entry_point: Some(Cow::from("init")),
        ..default()
    });
    let update_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![texture_bind_group_layout.clone()],
        shader: shader.clone(),
        entry_point: Some(Cow::from("update")),
        ..default()
    });
    let diffuse_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![texture_bind_group_layout.clone()],
        shader,
        entry_point: Some(Cow::from("diffuse")),
        ..default()
    });

    commands.insert_resource(FluidSimPipeline {
        texture_bind_group_layout,
        init_pipeline,
        update_pipeline,
        diffuse_pipeline,
    });
}

#[derive(Resource, Default)]
enum FluidSimState {
    #[default]
    Loading,
    Init,
    Update(usize),
}

fn update(
    pipeline: Res<FluidSimPipeline>,
    pipeline_cache: Res<PipelineCache>,
    mut state: ResMut<FluidSimState>,
) {
    // if the corresponding pipeline has loaded, transition to the next stage
    match *state {
        FluidSimState::Loading => {
            match pipeline_cache.get_compute_pipeline_state(pipeline.init_pipeline) {
                CachedPipelineState::Ok(_) => {
                    *state = FluidSimState::Init;
                }
                // If the shader hasn't loaded yet, just wait.
                CachedPipelineState::Err(ShaderCacheError::ShaderNotLoaded(_)) => {}
                CachedPipelineState::Err(err) => {
                    panic!("Initializing assets/{SHADER_ASSET_PATH}:\n{err}")
                }
                _ => {}
            }
        }
        FluidSimState::Init => {
            if let CachedPipelineState::Ok(_) =
                pipeline_cache.get_compute_pipeline_state(pipeline.update_pipeline)
            {
                *state = FluidSimState::Update(1);
            }
        }
        FluidSimState::Update(0) => {
            *state = FluidSimState::Update(1);
        }
        FluidSimState::Update(1) => {
            *state = FluidSimState::Update(0);
        }
        FluidSimState::Update(_) => unreachable!(),
    }
}

fn fluid_simulation(
    mut render_context: RenderContext,
    bind_groups: Res<FluidSimBindGroups>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<FluidSimPipeline>,
    state: Res<FluidSimState>,
) {
    let mut pass = render_context
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor::default());

    match *state {
        FluidSimState::Loading => {}

        FluidSimState::Init => {
            let init_pipeline = pipeline_cache
                .get_compute_pipeline(pipeline.init_pipeline)
                .unwrap();

            pass.set_pipeline(init_pipeline);
            pass.set_bind_group(0, &bind_groups.0[0], &[]);

            pass.dispatch_workgroups(
                SIZE.x.div_ceil(WORKGROUP_SIZE),
                SIZE.y.div_ceil(WORKGROUP_SIZE),
                1,
            );
        }

        FluidSimState::Update(index) => {
            let update_pipeline = pipeline_cache
                .get_compute_pipeline(pipeline.update_pipeline)
                .unwrap();

            pass.set_pipeline(update_pipeline);
            pass.set_bind_group(0, &bind_groups.0[index], &[]);

            pass.dispatch_workgroups(
                SIZE.x.div_ceil(WORKGROUP_SIZE),
                SIZE.y.div_ceil(WORKGROUP_SIZE),
                1,
            );

            let diffuse_pipeline = pipeline_cache
                .get_compute_pipeline(pipeline.diffuse_pipeline)
                .unwrap();

            pass.set_pipeline(diffuse_pipeline);

            for i in 0..20 {
                let diffuse_index = (index + 1 + i) % 2;

                pass.set_bind_group(0, &bind_groups.0[diffuse_index], &[]);

                pass.dispatch_workgroups(
                    SIZE.x.div_ceil(WORKGROUP_SIZE),
                    SIZE.y.div_ceil(WORKGROUP_SIZE),
                    1,
                );
            }
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
