# geam-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME geam
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
    LINK_LIBRARIES cublas
)
