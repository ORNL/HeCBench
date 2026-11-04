# minisweep-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME minisweep
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
    COMPILE_OPTIONS -munsafe-fp-atomics
)
