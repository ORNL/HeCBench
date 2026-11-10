# logprob-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME logprob
    MODEL hip
    SOURCES main.cu
    COMPILE_OPTIONS -ffast-math
    CATEGORIES ml
)
