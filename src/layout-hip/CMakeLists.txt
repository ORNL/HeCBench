# layout-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME layout
    MODEL hip
    SOURCES main.cu
    CATEGORIES language
)
