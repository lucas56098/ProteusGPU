# ProteusGPU

[![Build](https://github.com/lucas56098/ProteusGPU/actions/workflows/build.yml/badge.svg)](https://github.com/lucas56098/ProteusGPU/actions/workflows/build.yml) [![Version 0.7](https://img.shields.io/badge/version-0.7-blue)](https://github.com/lucas56098/ProteusGPU/releases)

![Banner](/docs/figures/banner_v6.webp)

Proteus is a GPU accelerated moving mesh hydrodynamics code.

It combines the algorithmic approach of ["Meshless Voronoi on the GPU" [Ray et. al 2018]](https://doi.org/10.1145/3272127.3275092) with a moving mesh hydro solver similar to ["AREPO" [Springel 2010]](https://academic.oup.com/mnras/article/401/2/791/1147356) ported to GPU.

> [!NOTE]
> The current version runs **3D moving mesh hydrodynamics on NVIDIA GH200** as well as discrete NVIDIA GPUs. A multithreaded CPU version, 2D and static meshes are supported. Optimizations are ongoing.

This project is being done during my master's thesis, supervised by [Dylan Nelson](https://nelson.tng-project.org/), at the Institute of Theoretical Astrophysics, Heidelberg University.

## Getting started

1. After cloning the repo select your system in `Makefile.systype` (or add your own to the `Makefile`)
2. Configure compilation flags in `Config.sh` and build with `make`
3. Use a `create.py` script for IC generation and specify simulation parameters in `param.txt`
4. Run the simulation with
```bash
./ProteusGPU [./ics/param.txt] [restart_flag]
```
If the `restart_flag` is set the simulation continues from the last snapshot in the `output_folder`.

## Examples 
Convergence of acoustic wave: second-order <br>
<img src="/docs/figures/convergence.png" alt="Video" width="50%">

Sod's shock tube test compared to AREPO<br>
<img src="/docs/figures/sod_convergence.png" alt="Image" width="80%">

Taylor-Sedov blast wave (2D/3D):<br>
<img src="/docs/figures/sedov.png" alt="Image" width="49%"> <img src="/docs/figures/sedov_3D_64.png" alt="Image" width="48%">

Kelvin Helmholtz Instability ($51^2$ and $1024^2$)<br>
<img src="/docs/figures/kh51.gif" alt="Image" width="49%"> <img src="/docs/figures/kh1024.png" alt="Image" width="49%">

Some colliding clouds ($200^3$, see ics/create_cloud_crash.py)<br>
<img src="/docs/figures/cloud_collision_3D.png" alt="Image" width="100%">

Runtime comparsion of Proteus (GPU/CPU) compared to AREPO for a $100^3$ shock tube test.<br>
<img src="/docs/figures/runtime_comparison_gh200.png" alt="Image" width="60%"> <br>
Note: since Proteus does not support MPI/Multi-node yet, the runtime comparison estimates AREPOs runtime from a single core time to eliminate most of its MPI overhead. Additionally its worth to note that Proteus does not work well on inhomogenous seed distributions yet.

## Dependencies
- HDF5 (`libhdf5-dev` on Ubuntu, via Homebrew on macOS)
- CUDA Toolkit (for GPU mode, requires NVIDIA GPU)

## Roadmap

* v0.1 - kNN and Voronoi mesh construction (2D and 3D, CPU)
* v0.2 - first order FV hydro (static mesh, 2D and 3D, CPU)
* v0.3 - second order FV hydro (static mesh, 2D and 3D, CPU)
* v0.4 - moving mesh hydro (2D, CPU)
* v0.5 - moving mesh hydro (3D, CPU)
* v0.6 - GPU initial port (moving mesh hydro)
* v0.7 - single GPU optimization
* v0.8 - multi-GPU, single node
* v0.9 - multi-node MPI
* v1.0 - support for inhomogenous particle distributions

