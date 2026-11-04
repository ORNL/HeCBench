# perlin-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME perlin
    MODEL hip
    SOURCES main.cu noise.cu
    CATEGORIES imaging
)
