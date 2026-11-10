# gaussian-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME gaussian
    MODEL cuda
    SOURCES gaussianElim.cu utils.cu
    CATEGORIES math
)
