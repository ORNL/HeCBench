# chacha20-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME chacha20
    MODEL omp
    SOURCES main.cpp
    CATEGORIES cryptography
)
