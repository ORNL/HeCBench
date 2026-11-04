# voxelization-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME voxelization
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES imaging
)
