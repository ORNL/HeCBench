# jacobian-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME jacobian
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
)
