# f16sp-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME f16sp
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
    LINK_LIBRARIES cublas
)
