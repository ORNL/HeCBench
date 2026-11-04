# segment-reduce-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME segment-reduce
    MODEL cuda
    SOURCES main.cu
    CATEGORIES compression
)
