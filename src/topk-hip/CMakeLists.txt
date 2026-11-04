# topk-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME topk
    MODEL hip
    SOURCES main.cu
    CATEGORIES sorting
)
