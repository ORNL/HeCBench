# background-subtract-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME background-subtract
    MODEL cuda
    SOURCES main.cu
    CATEGORIES imaging
)
