# langevin-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME langevin
    MODEL hip
    SOURCES main.cu
    COMPILE_OPTIONS -ffast-math
    CATEGORIES math
)
