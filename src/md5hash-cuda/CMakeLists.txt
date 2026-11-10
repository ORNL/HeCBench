# md5hash-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME md5hash
    MODEL cuda
    SOURCES MD5Hash.cu
    CATEGORIES encoding
)
