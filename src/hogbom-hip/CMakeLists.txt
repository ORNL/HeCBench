# hogbom-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../hogbom-cuda")

add_hecbench_benchmark(
    NAME hogbom
    MODEL hip
    SOURCES kernels.cu
            ${INC_DIR}/main.cpp
            ${INC_DIR}/reference.cpp
            ${INC_DIR}/timer.cpp
    CATEGORIES imaging
)
