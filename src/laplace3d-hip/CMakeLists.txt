# laplace3d-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME laplace3d
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
)
