# rotary-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME rotary
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
