# spmm-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME spmm
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
    LINK_LIBRARIES cusparse
)
