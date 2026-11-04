# zoom-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME zoom
    MODEL hip
    SOURCES main.cu
    CATEGORIES imaging
)
