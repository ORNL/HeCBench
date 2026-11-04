# mcpr-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME mcpr
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
