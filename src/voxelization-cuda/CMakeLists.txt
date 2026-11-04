# voxelization-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME voxelization
    MODEL cuda
    SOURCES main.cu
    CATEGORIES imaging
)
