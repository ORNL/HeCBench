# crc64-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME crc64
    MODEL hip
    SOURCES CRC64.cu CRC64Test.cu
    CATEGORIES encoding
)
