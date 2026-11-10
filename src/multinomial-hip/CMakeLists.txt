# multinomial-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME multinomial
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
