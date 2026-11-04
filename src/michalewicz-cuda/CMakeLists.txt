# michalewicz-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME michalewicz
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
)
