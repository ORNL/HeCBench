# graphExecution-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME graphExecution
    MODEL hip
    SOURCES main.cu
    CATEGORIES language
)
