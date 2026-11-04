# lud-sycl/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../lud-cuda/common")

add_hecbench_benchmark(
    NAME lud
    MODEL sycl
    SOURCES lud.cpp ${INC_DIR}/common.cpp
    INCLUDE_DIRS ${INC_DIR}
    CATEGORIES math
)
