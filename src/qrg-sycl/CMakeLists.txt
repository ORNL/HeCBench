# qrg-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME qrg
    MODEL sycl
    SOURCES main.cpp reference.cpp
    CATEGORIES rng
)
