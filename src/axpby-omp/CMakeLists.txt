# axpby-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME axpby
    MODEL omp
    SOURCES main.cpp
    CATEGORIES algorithms
)
