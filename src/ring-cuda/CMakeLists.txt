# ring-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME ring
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
)
