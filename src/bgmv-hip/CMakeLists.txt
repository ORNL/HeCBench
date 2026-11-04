# bgmv-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME bgmv
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
)
