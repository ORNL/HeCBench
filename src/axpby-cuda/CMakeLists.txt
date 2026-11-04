# axpby-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME axpby
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
)
