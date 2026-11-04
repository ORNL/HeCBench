# mrc-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME mrc
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
