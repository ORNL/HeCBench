# chi2-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME chi2
    MODEL hip
    SOURCES chi2.cu
    CATEGORIES simulation
)
