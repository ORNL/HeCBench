# spgeam-cuda/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../spgeam-sycl")

add_hecbench_benchmark(
    NAME spgeam
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
    INCLUDE_DIRS ${INC_DIR}
    LINK_LIBRARIES cusparse
)
