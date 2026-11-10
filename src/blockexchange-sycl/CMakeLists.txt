# blockexchange-sycl/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../blockAccess-sycl")

add_hecbench_benchmark(
    NAME blockexchange
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES language
    INCLUDE_DIRS ${INC_DIR}
)
