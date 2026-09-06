use bevy::prelude::*;

#[derive(Copy, Clone, Resource)]
pub struct DisplayFactor(pub u32);

impl Default for DisplayFactor {
    fn default() -> Self {
        Self(1)
    }
}
