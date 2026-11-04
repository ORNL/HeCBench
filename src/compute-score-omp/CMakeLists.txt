# compute-score-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME compute-score
    MODEL omp
    SOURCES main.cpp options.cpp
    CATEGORIES compression
)
