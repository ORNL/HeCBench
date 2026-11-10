# keccaktreehash-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME keccaktreehash
    MODEL omp
    SOURCES KeccakF.cpp KeccakTreeCPU.cpp KeccakTreeGPU.cpp Test.cpp main.cpp
    CATEGORIES cryptography
)
