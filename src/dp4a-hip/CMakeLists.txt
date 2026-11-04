# dp4a-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME dp4a
    MODEL hip
    SOURCES main.cu
    COMPILE_OPTIONS -DCDNA
    CATEGORIES language
)
