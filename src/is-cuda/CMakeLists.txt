# is-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME is
    MODEL cuda
    SOURCES main.cu
    CATEGORIES sorting
)
