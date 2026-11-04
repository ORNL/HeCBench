# memtest-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME memtest
    MODEL cuda
    SOURCES main.cu
    CATEGORIES bandwidth
)
