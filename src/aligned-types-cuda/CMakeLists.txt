# aligned-types-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME aligned-types
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
)
