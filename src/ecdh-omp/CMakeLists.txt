# ecdh-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME ecdh
    MODEL omp
    SOURCES main.cpp
    CATEGORIES cryptography
)
