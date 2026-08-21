# ldpc-sycl/CMakeLists.txt

set(CPU_REFERENCE "${CMAKE_CURRENT_LIST_DIR}/../ldpc-cuda/cpu.cpp")

add_hecbench_benchmark(
    NAME ldpc
    MODEL sycl
    SOURCES ${CPU_REFERENCE} main.cpp
    CATEGORIES algorithms
)
