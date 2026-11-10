# mnist-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME mnist
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
