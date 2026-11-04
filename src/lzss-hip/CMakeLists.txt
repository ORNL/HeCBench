# lzss-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME lzss
    MODEL hip
    SOURCES main.cu
    CATEGORIES compression
)
