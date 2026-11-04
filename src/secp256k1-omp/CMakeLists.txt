# secp256k1-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME secp256k1
    MODEL omp
    SOURCES main.cpp
    CATEGORIES math
)
