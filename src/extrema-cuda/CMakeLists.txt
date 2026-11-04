# extrema-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME extrema
    MODEL cuda
    SOURCES main.cu
    CATEGORIES signal
)
