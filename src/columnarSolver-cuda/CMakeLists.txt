# columnarSolver-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME columnarSolver
    MODEL cuda
    SOURCES main.cu
    CATEGORIES cryptography
)
