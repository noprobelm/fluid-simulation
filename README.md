# Fluid Dynamics Simulation

This project uses Bevy compute shaders to render a fluid dynamics simulation based on Jos Stam's paper: *Real-Time Fluid Dynamics for Games* (original paper available [here](https://graphics.cs.cmu.edu/nsp/course/15-464/Fall09/papers/StamFluidforGames.pdf))

Our implementation differs slightly from Stam's in that we are using compute shaders, as opposed to their CPU implementation. As such, we use textures to manage the density, vleocity, pressure, and divergence grids. We also use Jacobi iteration for diffusion and pressure solutions, as this felt more conducive to a compute shader approach than Stam's Guass-Seidel relaxation approach.

## Demo

https://github.com/user-attachments/assets/5c448e9b-d733-4c87-a0c7-c561ed170309
