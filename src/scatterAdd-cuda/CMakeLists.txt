# scatterAdd-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME scatterAdd
    MODEL cuda
    SOURCES main.cu
    CATEGORIES compression
)
