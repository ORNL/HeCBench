# scan3-sycl/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../scan2-sycl")

add_hecbench_benchmark(
    NAME scan3
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES compression
    INCLUDE_DIRS ${INC_DIR}
)
