# murmurhash3-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME murmurhash3
    MODEL omp
    SOURCES murmurhash3.cpp
    CATEGORIES encoding
)
