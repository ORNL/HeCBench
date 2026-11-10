# keccaktreehash-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME keccaktreehash
    MODEL sycl
    SOURCES KeccakF.cpp KeccakTreeCPU.cpp KeccakTreeGPU.cpp Test.cpp main.cpp
    CATEGORIES cryptography
)
