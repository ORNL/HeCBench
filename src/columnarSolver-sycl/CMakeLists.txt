# columnarSolver-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME columnarSolver
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES cryptography
)
