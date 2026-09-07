use bevy::{
    material::descriptor::{BindGroupLayoutDescriptor, CachedComputePipelineId},
    prelude::*,
    render::{extract_resource::ExtractResource, render_resource::BindGroup},
};

#[derive(Copy, Clone, Eq, PartialEq, Resource)]
pub struct DisplayFactor(pub u32);

impl Default for DisplayFactor {
    fn default() -> Self {
        Self(1)
    }
}

#[derive(Copy, Clone, ExtractResource, Resource)]
pub struct Iterations {
    pub density_diffusion: usize,
    pub velocity_diffusion: usize,
    pub pressure_solve: usize,
}

impl Default for Iterations {
    fn default() -> Self {
        Self {
            density_diffusion: 10,
            velocity_diffusion: 10,
            pressure_solve: 10,
        }
    }
}

#[derive(Copy, Clone, Default)]
pub enum PingPong {
    #[default]
    A,
    B,
}

#[derive(Resource, Default)]
pub enum FluidSimState {
    #[default]
    Loading,
    Init,
    Update,
}

impl PingPong {
    pub fn index(self) -> usize {
        match self {
            Self::A => 0,
            Self::B => 1,
        }
    }

    pub fn swap(&mut self) {
        *self = match self {
            Self::A => Self::B,
            Self::B => Self::A,
        };
    }
}

#[derive(Clone, Resource, Default)]
pub struct FluidSimBuffers {
    pub density: PingPong,
    pub dye: PingPong,
    pub velocity: PingPong,
    pub pressure: PingPong,
    pub reset_generation: u64,
}

#[derive(Resource, Clone, ExtractResource)]
pub struct FluidSimImages {
    pub density_original: Handle<Image>,
    pub density_current: Handle<Image>,
    pub density_next: Handle<Image>,
    pub dye_original: Handle<Image>,
    pub dye_current: Handle<Image>,
    pub dye_next: Handle<Image>,
    pub velocity_original: Handle<Image>,
    pub velocity_current: Handle<Image>,
    pub velocity_next: Handle<Image>,
    pub divergence: Handle<Image>,
    pub pressure_current: Handle<Image>,
    pub pressure_next: Handle<Image>,
    pub display: Handle<Image>,
}

#[derive(Resource)]
pub struct FluidSimPipeline {
    pub init_density_dye_bind_group_layout: BindGroupLayoutDescriptor,
    pub init_velocity_display_bind_group_layout: BindGroupLayoutDescriptor,
    pub add_sources_bind_group_layout: BindGroupLayoutDescriptor,
    pub density_diffuse_bind_group_layout: BindGroupLayoutDescriptor,
    pub velocity_diffuse_bind_group_layout: BindGroupLayoutDescriptor,
    pub velocity_advect_bind_group_layout: BindGroupLayoutDescriptor,
    pub density_advect_bind_group_layout: BindGroupLayoutDescriptor,
    pub compute_divergence_bind_group_layout: BindGroupLayoutDescriptor,
    pub divergence_set_bnd_bind_group_layout: BindGroupLayoutDescriptor,
    pub pressure_clear_bind_group_layout: BindGroupLayoutDescriptor,
    pub pressure_solve_bind_group_layout: BindGroupLayoutDescriptor,
    pub velocity_project_bind_group_layout: BindGroupLayoutDescriptor,
    pub density_visualize_bind_group_layout: BindGroupLayoutDescriptor,

    pub init_density_dye_pipeline: CachedComputePipelineId,
    pub init_velocity_display_pipeline: CachedComputePipelineId,
    pub add_sources_pipeline: CachedComputePipelineId,
    pub density_diffuse_pipeline: CachedComputePipelineId,
    pub velocity_diffuse_pipeline: CachedComputePipelineId,
    pub velocity_advect_pipeline: CachedComputePipelineId,
    pub density_advect_pipeline: CachedComputePipelineId,
    pub compute_divergence_pipeline: CachedComputePipelineId,
    pub divergence_set_bnd_pipeline: CachedComputePipelineId,
    pub pressure_clear_pipeline: CachedComputePipelineId,
    pub pressure_solve_pipeline: CachedComputePipelineId,
    pub velocity_project_pipeline: CachedComputePipelineId,
    pub density_visualize_pipeline: CachedComputePipelineId,
}

#[derive(Resource)]
pub struct FluidSimBindGroups {
    pub init_density_dye: BindGroup,
    pub init_velocity_display: BindGroup,

    // [density ping-pong state][velocity ping-pong state]
    pub add_sources: [[BindGroup; 2]; 2],
    pub density_diffuse: [BindGroup; 2],
    pub velocity_diffuse: [BindGroup; 2],
    pub velocity_advect: [BindGroup; 2],

    // [density ping-pong state][velocity ping-pong state]
    pub density_advect: [[BindGroup; 2]; 2],

    pub compute_divergence: [BindGroup; 2],
    pub divergence_set_bnd: [BindGroup; 2],

    // [pressure ping-pong state]
    pub pressure_clear: [BindGroup; 2],
    pub pressure_solve: [BindGroup; 2],

    // [velocity ping-pong state][pressure ping-pong state]
    pub velocity_project: [[BindGroup; 2]; 2],

    pub density_visualize: [BindGroup; 2],
}
