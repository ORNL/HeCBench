# mixbench-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME mixbench
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
)
