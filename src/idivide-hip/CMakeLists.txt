# idivide-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME idivide
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
)
