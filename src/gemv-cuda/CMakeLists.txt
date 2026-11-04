# gemv-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME gemv
    MODEL cuda
    SOURCES gemv.cu main.cu
    CATEGORIES math
)
