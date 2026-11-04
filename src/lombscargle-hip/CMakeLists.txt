# lombscargle-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME lombscargle
    MODEL hip
    SOURCES main.cu
    CATEGORIES signal
)
