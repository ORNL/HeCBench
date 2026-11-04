# tensorAccessor-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME tensorAccessor
    MODEL hip
    SOURCES main.cu
    CATEGORIES language
)
