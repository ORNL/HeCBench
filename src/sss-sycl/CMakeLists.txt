# sss-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME sss
    MODEL sycl
    SOURCES main.cpp
    INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}/../sss-cuda
    CATEGORIES algorithms
)
