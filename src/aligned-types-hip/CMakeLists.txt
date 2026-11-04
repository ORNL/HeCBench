# aligned-types-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME aligned-types
    MODEL hip
    SOURCES main.cu
    CATEGORIES language
)
