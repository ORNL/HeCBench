# CMake Build System for HeCBench

This document describes the CMake-based build system for HeCBench. The initial system was created by Jeffrey Vetter and Claude. 

## Overview

The CMake build system provides:
- **Unified configuration** via CMake presets for common GPU architectures
- **Selective building** by benchmark, programming model, or category
- **Automatic compiler detection** for CUDA, HIP, SYCL, and OpenMP
- **Parallel builds** across benchmarks
- **IDE integration** (CLion, VS Code, etc.)

## Quick Start

### Prerequisites

Depending on which programming models you want to build:

- **CUDA**: NVIDIA CUDA Toolkit (12.2+)
- **HIP**: AMD ROCm (7.0+)
- **SYCL**: Intel oneAPI DPC++ or hipSYCL
- **OpenMP**: Intel oneAPI, NVIDIA HPC SDK, or AOMP
- **CMake**: 3.21 or later
- **Ninja** (at least 1.10) or Make

### Build with a Preset

The simplest way to build is using a CMake preset:

```bash
# List available presets
cmake --list-presets

# Configure for building CUDA programs targeting NVIDIA Hopper GPUs with NVIDIA HPC SDK (version 25.7)
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
      -DMPI_C_COMPILER=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/comm_libs/mpi/bin/mpicc \
      -DMPI_CXX_COMPILER=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/comm_libs/mpi/bin/mpicxx \
      -DCUDAToolkit_ROOT=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/cuda/12.9/ \
      --preset cuda-sm90

# Build all configured benchmarks
cmake --build build/cuda-sm90

# Or build with Ninja in parallel
cmake --build build/cuda-sm90 --parallel

# Configure for building OpenMP offloading programs targeting AMD GFX942 GPUs with amdclang++
cmake -DCMAKE_BUILD_TYPE=Release \
      --preset openmp-amd-gfx942

```

### Available Presets

#### NVIDIA GPUs (CUDA)
- `cuda-sm60`  - Pascal (GTX 1080, P100)
- `cuda-sm70`  - Volta (V100)
- `cuda-sm80`  - Ampere (A100)
- `cuda-sm90`  - Hopper (H100, H200)
- `cuda-sm120` - Blackwell
- `cuda-sm121` - Blackwell (GB10)

#### AMD GPUs (HIP)
- `hip-gfx908`  - MI100
- `hip-gfx90a`  - MI210, MI250X
- `hip-gfx942`  - MI300A/X
- `hip-gfx1012` - Radeon RX 5500
- `hip-gfx1030` - Radeon RX 6900

#### SYCL
- `sycl-cuda-sm70` - SYCL with CUDA backend targeting Volta (Experimental)
- `sycl-cuda-sm80` - SYCL with CUDA backend targeting Ampere (Experimental)
- `sycl-cuda-sm90` - SYCL with CUDA backend targeting Hopper (Experimental)
- `sycl-hip-gfx908` - SYCL with HIP backend targeting MI100 (Experimental)
- `sycl-hip-gfx90a` - SYCL with HIP backend targeting MI210, MI250X (Experimental)
- `sycl-hip-gfx942` - SYCL with HIP backend targeting MI300A/X (Experimental)
- `sycl-cpu` - SYCL with CPU backend
- `sycl-xpu` - SYCL with XPU backend (Intel GPUs)

#### OpenMP offload
- `openmp-intel` - Intel compiler with OpenMP offload
- `openmp-nvidia-sm70` - NVIDIA compiler with OpenMP offload to Volta
- `openmp-nvidia-sm80` - NVIDIA compiler with OpenMP offload to Ampere
- `openmp-nvidia-sm90` - NVIDIA compiler with OpenMP offload to Hopper
- `openmp-amd-gfx908` - AMD compiler with OpenMP offload to MI100
- `openmp-amd-gfx90a` - AMD compiler with OpenMP offload to MI210, MI250X
- `openmp-amd-gfx942` - AMD compiler with OpenMP offload to MI300A/X

#### Multi-Model
- `all-models` - Build all programming models (requires all compilers)

## Building Specific Benchmarks

### Build a Single Benchmark (All Models)

```bash
cmake -DCMAKE_BUILD_TYPE=Release --preset all-models
cmake --build build/all-models --target jacobi-all
```

This builds jacobi for all enabled models (if you used `all-models` preset).

### Build a Specific Model Variant

```bash
# Build only CUDA version of jacobi
cmake -DCMAKE_BUILD_TYPE=Release --preset cuda-sm80
cmake --build build/cuda-sm80 --target jacobi-cuda

# Build only HIP version of attention
cmake -DCMAKE_BUILD_TYPE=Release --preset hip-gfx90a
cmake --build build/hip-gfx90a --target attention-hip

# Build only SYCL XPU version of attention
source /opt/intel/oneapi/setvars.sh
cmake -DCMAKE_BUILD_TYPE=Release --preset sycl-xpu
cmake --build build/sycl-xpu --target attention-sycl
```

### Build by Category

```bash
# Build all machine learning benchmarks
cmake --build build/cuda-sm80 --target category-ml

# Build all graph benchmarks
cmake --build build/cuda-sm80 --target category-graph

# Build all simulation benchmarks
cmake --build build/cuda-sm80 --target category-simulation
```

## Advanced Configuration

### Custom Configuration

If you need more control, configure without a preset:

```bash
cmake -B build/custom \
  -G Ninja \
  -DHECBENCH_ENABLE_CUDA=ON \
  -DHECBENCH_ENABLE_HIP=OFF \
  -DHECBENCH_ENABLE_SYCL=OFF \
  -DHECBENCH_ENABLE_OPENMP=OFF \
  -DHECBENCH_CUDA_ARCH=80

cmake --build build/custom
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `HECBENCH_ENABLE_CUDA` | ON | Enable CUDA benchmarks |
| `HECBENCH_ENABLE_HIP` | ON | Enable HIP benchmarks |
| `HECBENCH_ENABLE_SYCL` | ON | Enable SYCL benchmarks |
| `HECBENCH_ENABLE_OPENMP` | ON | Enable OpenMP benchmarks |
| `HECBENCH_CUDA_ARCH` | sm_80 | CUDA architecture (60, 70, 80, 90, etc.) |
| `HECBENCH_HIP_ARCH` | gfx90a | HIP architecture (gfx908, gfx90a, gfx942, etc.) |
| `HECBENCH_SYCL_TARGET` | (auto) | SYCL target backend |
| `HECBENCH_ENABLE_TESTING` | ON | Enable testing support |
| `HECBENCH_BUILD_ALL_BENCHMARKS` | ON | Build all vs. selective |

### Multi-Architecture Builds

To build for multiple GPU architectures, use multiple configure+build cycles:

```bash
# Build for A100
cmake -DCMAKE_BUILD_TYPE=Release --preset cuda-sm80
cmake --build build/cuda-sm80

# Build for V100
cmake -DCMAKE_BUILD_TYPE=Release --preset cuda-sm70
cmake --build build/cuda-sm70

# Build for MI250X
cmake -DCMAKE_BUILD_TYPE=Release --preset hip-gfx90a
cmake --build build/hip-gfx90a
```

## Output Structure

Compiled binaries are placed in:

```
build/<preset>/bin/<model>/
├── cuda/
│   ├── jacobi
│   ├── bfs
│   ├── softmax
│   └── ...
├── hip/
│   └── ...
├── sycl/
│   └── ...
└── omp/
    └── ...
```

## Running Benchmarks

```bash
# Run a benchmark directly
./build/cuda-sm80/bin/cuda/jacobi

# Run with specific GPU
CUDA_VISIBLE_DEVICES=0 ./build/cuda-sm80/bin/cuda/attention
```

### Benchmarks Not Yet Converted

The following benchmarks do not support CMake build.

| Benchmark | Variants | Reason |
|-----------|----------|--------|
| `convolutioneformable` | cuda, hip, sycl | Python/PyTorch extension (setup.py build) |
| `dwconv1d` | cuda, hip, sycl | Python/PyTorch extension (run.py build) |
| `dp` | cuda | Requires nvc++ compiler |
| `hpl` | cuda, hip, sycl | complex workflow |
| `miniDGS` | cuda | MPI + ParMetis dependency |
| `saxpy-ompt` | cuda, hip | Requires nvc++ compiler and amdclang++ (OpenMP target offload) |

These benchmarks still work with their original Makefiles.

## Adding New Benchmarks

To convert a benchmark to CMake, add a `CMakeLists.txt` in each model directory:

```cmake
# src/mybench-cuda/CMakeLists.txt
add_hecbench_benchmark(
    NAME mybench
    MODEL cuda
    SOURCES main.cu kernel.cu
    CATEGORIES simulation
)
```

### Available Options

```cmake
add_hecbench_benchmark(
    NAME mybench                    # Benchmark name (required)
    MODEL cuda                      # Programming model: cuda, hip, sycl, omp (required)
    SOURCES main.cu kernel.cu       # Source files (required)
    CATEGORIES simulation           # One README category (optional)
    INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include  # Additional include paths (optional)
    COMPILE_OPTIONS -maxrregcount=32                  # Extra compiler flags (optional)
    LINK_LIBRARIES CUDA::cublas                       # Libraries to link (optional)
)
```

## Troubleshooting

### Compiler Not Found

```
CMake Error: Could not find CUDA/HIP/SYCL compiler
```

**Solution**: Install the required compiler or disable that model:
```bash
cmake -DCMAKE_BUILD_TYPE=Release --preset cuda-sm80 -DHECBENCH_ENABLE_HIP=OFF
```

### Architecture Mismatch

```
Error: Unsupported architecture sm_XX
```

**Solution**: Use a preset matching your GPU or set the architecture manually:
```bash
cmake -DCMAKE_BUILD_TYPE=Release --preset cuda-sm80 -DHECBENCH_CUDA_ARCH=86
```

### Missing Dependencies

Some benchmarks may require additional libraries (oneDPL, TBB, cuFFT, cuBLAS, etc.). These will be detected automatically if present.

## IDE Integration

### Visual Studio Code

Install the CMake Tools extension, then:
1. Open the HeCBench folder
2. Select a CMake preset from the status bar
3. Click "Build" or press F7

### CLion

CLion automatically detects `CMakePresets.json`:
1. Open the HeCBench project
2. CLion will import presets automatically
3. Select a profile from the dropdown
4. Build → Build Project

## Comparison with Makefile Build

| Feature | Makefile | CMake |
|---------|----------|-------|
| Configuration | Edit 2,587 individual Makefiles | Single preset selection |
| Parallel builds | Per-benchmark only | Across all benchmarks |
| Selective building | Manual (cd + make) | Target-based (by name/category) |
| IDE support | Limited | Full integration |
| Multi-arch | Rebuild everything | Separate build dirs |
| Dependency tracking | Manual | Automatic |

## Future Enhancements

Planned improvements:
- [x] Migrate benchmarks to CMake (98% complete)
- [ ] Convert remaining 11 complex benchmarks
- [ ] CTest integration for automated testing
- [ ] CPack support for distribution
- [ ] Benchmark performance regression tracking
- [ ] Docker container presets
- [ ] GitHub Actions CI integration

## Getting Help

- Report issues: https://github.com/zjin-lcf/HeCBench/issues
- Main README: [README.md](README.md)
- Full renovation plan: [plan.md](plan.md)

---

**Last Updated**: 2025-12-07
**Status**: Phase 2 Nearly Complete (98% migration)
