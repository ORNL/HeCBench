# columnarSolver-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME columnarSolver
    MODEL omp
    SOURCES main.cpp
    CATEGORIES cryptography
)
