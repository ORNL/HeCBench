# aop-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME aop
    MODEL hip
    SOURCES main.cu
    CATEGORIES finance
)
