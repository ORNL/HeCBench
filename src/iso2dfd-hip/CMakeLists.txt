# iso2dfd-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME iso2dfd
    MODEL hip
    SOURCES iso2dfd.cu
    CATEGORIES simulation
)
