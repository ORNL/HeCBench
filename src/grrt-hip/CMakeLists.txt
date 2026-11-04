# grrt-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME grrt
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
    COMPILE_OPTIONS -ffast-math
)
