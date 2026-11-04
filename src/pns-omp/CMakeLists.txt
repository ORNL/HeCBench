# pns-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME pns
    MODEL omp
    SOURCES main.cpp
    CATEGORIES simulation
)
