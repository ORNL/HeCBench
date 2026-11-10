# merkle-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME merkle
    MODEL sycl
    SOURCES bench_merkle_tree.cpp ff_p.cpp main.cpp merkle_tree.cpp rescue_prime.cpp
    CATEGORIES cryptography
)
