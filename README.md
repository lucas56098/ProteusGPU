# ProteusGPU

[![Build and Test](https://github.com/lucas56098/ProteusGPU/actions/workflows/build.yml/badge.svg)](https://github.com/lucas56098/ProteusGPU/actions/workflows/build.yml) [![Version 0.0.1](https://img.shields.io/badge/version-0.0.1-blue)](https://github.com/lucas56098/ProteusGPU/releases)

![Banner](/figures/banner_v3.jpg)

Work in progress: A GPU accelerated moving mesh hydrodynamics code.

Dont expect anything to work yet for the forseeable future.

The idea is to combine: ["Meshless Voronoi on the GPU" [Ray et. al 2018]](https://doi.org/10.1145/3272127.3275092) with a moving mesh hydro solver similar to ["AREPO" [Springel 2010]](https://academic.oup.com/mnras/article/401/2/791/1147356) ported to GPU. Doing this in a toy code allows to explore various ideas first.

This project is being done during my master's thesis, supervised by Dylan Nelson, at the Institute of Theoretical Astrophysics, Heidelberg University.

## Building

1. Select your system in `Makefile.systype` (Ubuntu/macOS, or add your own)
2. Configure compilation flags in `Config.sh` and parameters in `param.txt`
3. Build with `make`

```bash
make           # Uses SYSTYPE from Makefile.systype
./ProteusGPU   # Run the executable
```

## Dependencies
- HDF5 (libhdf5-dev on Ubuntu, via Homebrew on macOS)