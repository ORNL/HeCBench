# megablocks-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME megablocks
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
    LINK_LIBRARIES cublas
)
