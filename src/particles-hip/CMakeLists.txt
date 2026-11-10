# particles-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME particles
    MODEL hip
    SOURCES bitonicSort.cu main.cu particles.cu
    CATEGORIES simulation
)
