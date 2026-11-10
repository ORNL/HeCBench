# merge-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME merge
    MODEL hip
    SOURCES main.cu
    CATEGORIES sorting
)
