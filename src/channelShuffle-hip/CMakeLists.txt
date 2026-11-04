# channelShuffle-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME channelShuffle
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
