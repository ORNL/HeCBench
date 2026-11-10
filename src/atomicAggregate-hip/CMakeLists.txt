# atomicAggregate-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME atomicAggregate
    MODEL hip
    SOURCES main.cu
    CATEGORIES compression
)
