# merkle-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME merkle
    MODEL cuda
    SOURCES bench_merkle_tree.cu ff_p.cu main.cu merkle_tree.cu rescue_prime.cu
    CATEGORIES cryptography
    COMPILE_OPTIONS --expt-relaxed-constexpr
)
