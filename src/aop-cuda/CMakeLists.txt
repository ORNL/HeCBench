# aop-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME aop
    MODEL cuda
    SOURCES main.cu
    CATEGORIES finance
)
