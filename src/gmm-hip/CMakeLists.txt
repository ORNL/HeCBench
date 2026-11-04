# gmm-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME gmm
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
