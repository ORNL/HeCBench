# gpp-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME gpp
    MODEL omp
    SOURCES main.cpp
    CATEGORIES simulation
)
