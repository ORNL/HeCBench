# spsm-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME spsm
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
    LINK_LIBRARIES cusparse
)
