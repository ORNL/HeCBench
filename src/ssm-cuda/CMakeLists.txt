# ssm-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME ssm
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
