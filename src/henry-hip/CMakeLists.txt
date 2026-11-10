# henry-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME henry
    MODEL hip
    SOURCES main.cu
    COMPILE_OPTIONS -ffast-math
    CATEGORIES simulation
)
