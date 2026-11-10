# chi2-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME chi2
    MODEL omp
    SOURCES chi2.cpp
    CATEGORIES simulation
)
