# doh-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME doh
    MODEL hip
    SOURCES main.cu
    CATEGORIES imaging
)
