# atomicPerf-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME atomicPerf
    MODEL hip
    SOURCES main.cu
    CATEGORIES compression
    COMPILE_OPTIONS -munsafe-fp-atomics
)
