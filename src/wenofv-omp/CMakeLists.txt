# wenofv-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME wenofv
    MODEL omp
    SOURCES main.cpp
    CATEGORIES algorithms
)
