# romberg-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME romberg
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
)
