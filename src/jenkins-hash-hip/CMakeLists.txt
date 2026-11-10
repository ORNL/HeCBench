# jenkins-hash-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME jenkins-hash
    MODEL hip
    SOURCES main.cu
    CATEGORIES encoding
)
