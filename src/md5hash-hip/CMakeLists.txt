# md5hash-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME md5hash
    MODEL hip
    SOURCES MD5Hash.cu
    CATEGORIES encoding
)
