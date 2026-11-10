# mis-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME mis
    MODEL cuda
    SOURCES main.cu
    CATEGORIES graph
)
