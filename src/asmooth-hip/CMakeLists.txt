# asmooth-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME asmooth
    MODEL hip
    SOURCES main.cu
    CATEGORIES imaging
    COMPILE_OPTIONS -munsafe-fp-atomics
)
