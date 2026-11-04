# fdtd3d-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME fdtd3d
    MODEL cuda
    SOURCES FDTD3dGPU.cu FDTD3dReference.cu cmd_arg_reader.cu main.cu shrUtils.cu
    CATEGORIES simulation
)
