# sort-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME sort
    MODEL hip
    SOURCES main.cu
    CATEGORIES sorting
)
