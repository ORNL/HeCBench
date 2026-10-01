# ldpc-hip/CMakeLists.txt

set(CPU_REFERENCE "${CMAKE_CURRENT_LIST_DIR}/../ldpc-cuda/cpu.cpp")

add_hecbench_benchmark(
    NAME ldpc
    MODEL hip
    SOURCES ${CPU_REFERENCE} main.cu
    CATEGORIES algorithms
)
