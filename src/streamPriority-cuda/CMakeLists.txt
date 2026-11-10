# streamPriority-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME streamPriority
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
)
