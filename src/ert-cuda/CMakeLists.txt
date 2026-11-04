# ert-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME ert
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
)
