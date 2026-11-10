# hbc-sycl/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../hbc-cuda")

add_hecbench_benchmark(
    NAME hbc
    MODEL sycl
    SOURCES kernels.cpp main.cpp
            ${INC_DIR}/parse.cpp ${INC_DIR}/sequential.cpp ${INC_DIR}/util.cpp
    CATEGORIES graph
)
