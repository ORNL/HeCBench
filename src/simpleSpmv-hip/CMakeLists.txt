# simpleSpmv-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../simpleSpmv-cuda")

add_hecbench_benchmark(
    NAME simpleSpmv
    MODEL hip
    SOURCES kernels.cu ${INC_DIR}/main.cpp ${INC_DIR}/utils.cpp
    CATEGORIES math
)
