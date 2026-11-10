# crc64-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME crc64
    MODEL cuda
    SOURCES CRC64.cu CRC64Test.cu
    CATEGORIES encoding
)
