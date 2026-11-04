# kurtosis-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME kurtosis
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
)
