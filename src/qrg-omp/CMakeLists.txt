# qrg-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME qrg
    MODEL omp
    SOURCES main.cpp reference.cpp
    CATEGORIES rng
)
