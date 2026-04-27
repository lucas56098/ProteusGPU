# ProteusGPU

[![Build](https://github.com/lucas56098/ProteusGPU/actions/workflows/build.yml/badge.svg)](https://github.com/lucas56098/ProteusGPU/actions/workflows/build.yml) [![Version 0.6](https://img.shields.io/badge/version-0.6-blue)](https://github.com/lucas56098/ProteusGPU/releases)

![Banner](/figures/banner_v5.webp)

Proteus is a GPU accelerated moving mesh hydrodynamics code.

It combines the algorithmic approach of ["Meshless Voronoi on the GPU" [Ray et. al 2018]](https://doi.org/10.1145/3272127.3275092) with a moving mesh hydro solver similar to ["AREPO" [Springel 2010]](https://academic.oup.com/mnras/article/401/2/791/1147356) ported to GPU.

> [!NOTE]
> The current version runs 2D/3D static/moving mesh hydrodynamics multithreaded on CPU, as well as on a single NVIDIA GPU. Optimizations are ongoing.

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

