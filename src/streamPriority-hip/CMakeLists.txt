# streamPriority-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME streamPriority
    MODEL hip
    SOURCES main.cu
    CATEGORIES language
)
