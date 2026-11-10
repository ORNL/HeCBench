# md5hash-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME md5hash
    MODEL sycl
    SOURCES MD5Hash.cpp
    CATEGORIES encoding
)
