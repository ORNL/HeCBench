# hypterm-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../hypterm-cuda")

add_hecbench_benchmark(
    NAME hypterm
    MODEL hip
    SOURCES kernels.cu ${INC_DIR}/main.cpp ${INC_DIR}/reference.cpp
    CATEGORIES simulation
)
