# wenofv-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME wenofv
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
