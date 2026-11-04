# overlay-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME overlay
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
