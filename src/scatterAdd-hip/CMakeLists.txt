# scatterAdd-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME scatterAdd
    MODEL hip
    SOURCES main.cu
    CATEGORIES compression
    COMPILE_OPTIONS -munsafe-fp-atomics
)
