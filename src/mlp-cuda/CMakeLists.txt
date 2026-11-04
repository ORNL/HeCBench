# mlp-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME mlp
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
    LINK_LIBRARIES cublasLt
)
