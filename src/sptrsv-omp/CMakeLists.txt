# sptrsv-omp/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../sptrsv-sycl")

add_hecbench_benchmark(
    NAME sptrsv
    MODEL omp
    SOURCES sptrsv_syncfree.cpp ${INC_DIR}/main.cpp
    CATEGORIES math
    COMPILE_OPTIONS -DVALUE_TYPE=double
    INCLUDE_DIRS ${INC_DIR}
)
