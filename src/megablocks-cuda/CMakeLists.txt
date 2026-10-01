# megablocks-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME megablocks
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
    LINK_LIBRARIES cublas
)
