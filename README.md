# Fluid Dynamics Simulation

This project uses Bevy compute shaders to render a fluid dynamics simulation based on Jos Stam's paper: *Real-Time Fluid Dynamics for Games* (original paper available [here](https://graphics.cs.cmu.edu/nsp/course/15-464/Fall09/papers/StamFluidforGames.pdf))

Our implementation differs slightly from Stam's in that we are using compute shaders, as opposed to their CPU implementation. As such, we use textures to manage the density, vleocity, pressure, and divergence grids. We also use Jacobi iteration for diffusion and pressure solutions, as this felt more conducive to a compute shader approach than Stam's Guass-Seidel relaxation approach.

## Demo

[![Fluid simulation demo](assets/demo/fluid_sim.gif)](assets/demo/fluid_sim.mp4)

Click the preview to watch the full-resolution video.
