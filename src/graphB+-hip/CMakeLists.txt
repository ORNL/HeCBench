# graphB+-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME graphB+
    MODEL hip
    SOURCES main.cu
    CATEGORIES graph
)
