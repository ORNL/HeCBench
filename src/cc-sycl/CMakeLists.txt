# cc-sycl/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../mis-cuda")

add_hecbench_benchmark(
    NAME cc
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES graph
    INCLUDE_DIRS ${INC_DIR}
)
