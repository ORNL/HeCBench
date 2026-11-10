# fdtd3d-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME fdtd3d
    MODEL hip
    SOURCES FDTD3dGPU.cu FDTD3dReference.cu cmd_arg_reader.cu main.cu shrUtils.cu
    CATEGORIES simulation
)
