# floydwarshall-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME floydwarshall
    MODEL cuda
    SOURCES main.cu
    CATEGORIES graph
)
