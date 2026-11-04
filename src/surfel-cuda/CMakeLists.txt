# surfel-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME surfel
    MODEL cuda
    SOURCES main.cu
    CATEGORIES imaging
)
