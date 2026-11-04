# columnarSolver-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME columnarSolver
    MODEL hip
    SOURCES main.cu
    CATEGORIES cryptography
)
