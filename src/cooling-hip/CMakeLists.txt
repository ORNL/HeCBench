# cooling-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME cooling
    MODEL hip
    SOURCES main.cu
    COMPILE_OPTIONS -ffast-math
    CATEGORIES simulation
)
