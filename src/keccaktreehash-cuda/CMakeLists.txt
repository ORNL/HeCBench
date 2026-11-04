# keccaktreehash-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME keccaktreehash
    MODEL cuda
    SOURCES KeccakF.cu KeccakTreeCPU.cu KeccakTreeGPU.cu Test.cu main.cu
    CATEGORIES cryptography
)
