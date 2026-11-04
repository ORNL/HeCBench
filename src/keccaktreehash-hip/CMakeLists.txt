# keccaktreehash-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME keccaktreehash
    MODEL hip
    SOURCES KeccakF.cu KeccakTreeCPU.cu KeccakTreeGPU.cu Test.cu main.cu
    CATEGORIES cryptography
)
