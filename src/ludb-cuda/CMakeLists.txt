# ludb-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME ludb
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
    LINK_LIBRARIES cublas
)
