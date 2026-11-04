# channelSum-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME channelSum
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
