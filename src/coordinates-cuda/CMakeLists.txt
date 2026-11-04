# coordinates-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME coordinates
    MODEL cuda
    SOURCES main.cu
    CATEGORIES geoscience
)
