# gabor-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME gabor
    MODEL hip
    SOURCES main.cu
    COMPILE_OPTIONS -ffast-math
    CATEGORIES imaging
)
