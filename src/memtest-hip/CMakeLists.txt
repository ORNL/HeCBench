# memtest-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME memtest
    MODEL hip
    SOURCES main.cu
    CATEGORIES bandwidth
)
