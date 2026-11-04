# deredundancy-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME deredundancy
    MODEL hip
    SOURCES main.cu utils.cu
    CATEGORIES bioinformatics
)
