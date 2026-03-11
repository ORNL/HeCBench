# ssm-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME ssm
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
