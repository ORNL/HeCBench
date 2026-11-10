# crossEntropy-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME crossEntropy
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
