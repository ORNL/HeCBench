# zerocopy-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME zerocopy
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
)
