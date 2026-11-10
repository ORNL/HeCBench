# is-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME is
    MODEL hip
    SOURCES main.cu
    CATEGORIES sorting
)
