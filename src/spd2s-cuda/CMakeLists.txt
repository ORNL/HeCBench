# spd2s-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME spd2s
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
    LINK_LIBRARIES cusparse
)
