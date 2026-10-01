# wenofv-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME wenofv
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
