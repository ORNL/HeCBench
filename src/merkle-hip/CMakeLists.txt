# merkle-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME merkle
    MODEL hip
    SOURCES bench_merkle_tree.cu ff_p.cu main.cu merkle_tree.cu rescue_prime.cu
    CATEGORIES cryptography
    COMPILE_OPTIONS -fgpu-rdc
    LINK_LIBRARIES -fgpu-rdc --hip-link
)
