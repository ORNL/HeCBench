# murmurhash3-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME murmurhash3
    MODEL hip
    SOURCES murmurhash3.cu
    CATEGORIES encoding
)
