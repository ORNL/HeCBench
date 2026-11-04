# rsmt-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME rsmt
    MODEL cuda
    SOURCES main.cu
    CATEGORIES graph
)
