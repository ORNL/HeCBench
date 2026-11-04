# spsort-cuda/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../spsort-sycl")

add_hecbench_benchmark(
    NAME spsort
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
    INCLUDE_DIRS ${INC_DIR}
    LINK_LIBRARIES cusparse
)
