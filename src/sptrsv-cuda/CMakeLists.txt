# sptrsv-cuda/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../sptrsv-sycl")

add_hecbench_benchmark(
    NAME sptrsv
    MODEL cuda
    SOURCES sptrsv_syncfree.cu ${INC_DIR}/main.cpp
    CATEGORIES math
    INCLUDE_DIRS ${INC_DIR}
    COMPILE_OPTIONS -DWARP_SIZE=32 -DWARP_PER_BLOCK=8 -DVALUE_TYPE=double
    LINK_LIBRARIES cusparse
)
