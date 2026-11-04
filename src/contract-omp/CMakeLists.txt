# contract-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME contract
    MODEL omp
    SOURCES main.cpp
    CATEGORIES compression
)
