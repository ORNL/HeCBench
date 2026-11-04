# atomicCost-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME atomicCost
    MODEL hip
    SOURCES main.cu
    CATEGORIES compression
    COMPILE_OPTIONS -munsafe-fp-atomics
)
