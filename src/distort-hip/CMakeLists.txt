# distort-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME distort
    MODEL hip
    SOURCES distort.cu main.cu
    CATEGORIES imaging
)
