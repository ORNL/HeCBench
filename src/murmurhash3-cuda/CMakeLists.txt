# murmurhash3-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME murmurhash3
    MODEL cuda
    SOURCES murmurhash3.cu
    CATEGORIES encoding
)
