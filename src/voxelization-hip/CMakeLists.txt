# voxelization-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME voxelization
    MODEL hip
    SOURCES main.cu
    CATEGORIES imaging
)
