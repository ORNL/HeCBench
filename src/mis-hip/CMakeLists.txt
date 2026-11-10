# mis-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME mis
    MODEL hip
    SOURCES main.cu
    CATEGORIES graph
)
