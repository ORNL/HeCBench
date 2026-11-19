# flame-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME flame
    MODEL cuda
    SOURCES main.cu utils.cu
    CATEGORIES imaging
)
