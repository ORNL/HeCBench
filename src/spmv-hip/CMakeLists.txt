# spmv-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../spmv-cuda")

add_hecbench_benchmark(
    NAME spmv
    MODEL hip
    SOURCES kernels.cu ${INC_DIR}/utils.cpp  ${INC_DIR}/main.cpp 
    CATEGORIES math
    LINK_LIBRARIES hipsparse
)
