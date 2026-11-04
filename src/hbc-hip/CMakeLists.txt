# hbc-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../hbc-cuda")

add_hecbench_benchmark(
    NAME hbc
    MODEL hip
    SOURCES kernels.cu
            ${INC_DIR}/main.cpp ${INC_DIR}/parse.cpp ${INC_DIR}/sequential.cpp ${INC_DIR}/util.cpp
    CATEGORIES graph
)
