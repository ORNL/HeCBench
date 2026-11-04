# laplace-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME laplace
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
)
