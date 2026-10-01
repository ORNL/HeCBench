# segment-reduce2-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME segment-reduce2
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
