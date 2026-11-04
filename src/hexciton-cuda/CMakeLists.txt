# hexciton-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME hexciton
    MODEL cuda
    SOURCES main.cu reference.cu utils.cu
    CATEGORIES simulation
)
