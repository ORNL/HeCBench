# vote-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME vote
    MODEL hip
    SOURCES main.cu
    CATEGORIES language
)
