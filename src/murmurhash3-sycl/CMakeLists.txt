# murmurhash3-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME murmurhash3
    MODEL sycl
    SOURCES murmurhash3.cpp
    CATEGORIES encoding
)
