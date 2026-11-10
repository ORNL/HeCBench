# mdh-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME mdh
    MODEL hip
    SOURCES WKFUtils.cu main.cu
    CATEGORIES simulation
)
