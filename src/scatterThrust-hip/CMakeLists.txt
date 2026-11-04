# scatterThrust-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME scatterThrust
    MODEL hip
    SOURCES main.cu
    CATEGORIES compression
)
