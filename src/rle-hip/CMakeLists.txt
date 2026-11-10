# rle-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME rle
    MODEL hip
    SOURCES main.cu
    CATEGORIES compression
)
