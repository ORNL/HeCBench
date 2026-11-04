# perplexity-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME perplexity
    MODEL hip
    SOURCES main.cu
    COMPILE_OPTIONS -ffast-math
    CATEGORIES ml
)
