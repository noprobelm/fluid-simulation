# Fluid Dynamics Simulation

This project uses Bevy compute shaders to render a fluid dynamics simulation. The simulation is based on Jos Stam's paper: *Real-Time Fluid Dynamics for Games* (available [here](https://graphics.cs.cmu.edu/nsp/course/15-464/Fall09/papers/StamFluidforGames.pdf))

Our implementation differs slightly from Stam's in that we are using compute shaders as opposed to their CPU implementation. As such, our implementations differ in some respects. For example, I found it easier to use Jacobi iteration for diffusion and pressure solutions instead of Stam's Guass-Seidel relaxation approach.

Beyond this, I made adaptations where necessary to cater to a compute shader approach (simply, this pretty much means we use textures for velocity/density/pressure/divergence grids).

## Demo

<https://github.com/user-attachments/assets/5c448e9b-d733-4c87-a0c7-c561ed170309>

## How to use

- Left click: Add density
- Right click with mouse movement: Add velocity vectors

## Development Status

UI elements and simulation controls are currently being developed.
