# asta-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME asta
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
)
