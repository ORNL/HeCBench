# crc64-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME crc64
    MODEL sycl
    SOURCES CRC64.cpp CRC64Test.cpp
    CATEGORIES encoding
)
