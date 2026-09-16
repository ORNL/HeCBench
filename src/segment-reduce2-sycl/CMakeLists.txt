# segment-reduce2-sycl/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../segment-reduce2-cuda")

add_hecbench_benchmark(
    NAME segment-reduce2
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
    INCLUDE_DIRS ${INC_DIR}
)
