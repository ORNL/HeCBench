# tissue-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME tissue
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
)
