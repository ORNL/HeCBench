# sptrsv-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../sptrsv-sycl")

add_hecbench_benchmark(
    NAME sptrsv
    MODEL hip
    SOURCES sptrsv_syncfree.cu ${INC_DIR}/main.cpp
    INCLUDE_DIRS ${INC_DIR}
    COMPILE_OPTIONS -DWARP_SIZE=64 -DWARP_PER_BLOCK=8 -DVALUE_TYPE=double
    CATEGORIES math
)
