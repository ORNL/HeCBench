# sss-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME sss
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
