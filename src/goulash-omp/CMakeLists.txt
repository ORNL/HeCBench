# goulash-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME goulash
    MODEL omp
    SOURCES main.cpp
    CATEGORIES simulation
)
