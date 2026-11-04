# tonemapping-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME tonemapping
    MODEL hip
    SOURCES main.cu
    CATEGORIES imaging
)
