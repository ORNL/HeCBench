# crc64-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME crc64
    MODEL omp
    SOURCES CRC64.cpp CRC64Test.cpp
    CATEGORIES encoding
)
