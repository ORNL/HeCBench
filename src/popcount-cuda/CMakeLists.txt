# popcount-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME popcount
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
)
