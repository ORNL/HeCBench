# contract-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME contract
    MODEL cuda
    SOURCES main.cu
    CATEGORIES compression
)
