# idivide-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME idivide
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
)
