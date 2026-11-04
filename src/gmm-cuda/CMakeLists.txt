# gmm-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME gmm
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
