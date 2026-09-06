use bevy::{prelude::*, render::extract_resource::ExtractResource};

#[derive(Copy, Clone, Resource)]
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
            density_diffusion: 6,
            velocity_diffusion: 6,
            pressure_solve: 6,
        }
    }
}
