# md5hash-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME md5hash
    MODEL omp
    SOURCES MD5Hash.cpp
    CATEGORIES encoding
)
