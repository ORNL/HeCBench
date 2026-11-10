# base64e-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME base64e
    MODEL hip
    SOURCES main.cu
    CATEGORIES encoding
)
