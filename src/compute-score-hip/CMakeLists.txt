# compute-score-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME compute-score
    MODEL hip
    SOURCES main.cu options.cu
    CATEGORIES compression
)
