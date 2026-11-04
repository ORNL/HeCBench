# doh-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME doh
    MODEL cuda
    SOURCES main.cu
    CATEGORIES imaging
)
