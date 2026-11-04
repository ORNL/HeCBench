# langford-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME langford
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
)
