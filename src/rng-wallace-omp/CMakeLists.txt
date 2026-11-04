# rng-wallace-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME rng-wallace
    MODEL omp
    SOURCES main.cpp
    CATEGORIES rng
)
