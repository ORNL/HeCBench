# ldpc-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME ldpc
    MODEL cuda
    SOURCES cpu.cpp main.cu
    CATEGORIES algorithms
)
