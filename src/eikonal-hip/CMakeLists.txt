# eikonal-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../eikonal-cuda")

add_hecbench_benchmark(
    NAME eikonal
    MODEL hip
    SOURCES StructuredEikonal.cu fim.cu kernel.cu
            ${INC_DIR}/timer.cpp
            ${INC_DIR}/main.cpp
    CATEGORIES simulation
)
