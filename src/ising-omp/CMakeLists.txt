# ising-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME ising
    MODEL omp
    SOURCES main.cpp
    CATEGORIES simulation
)
