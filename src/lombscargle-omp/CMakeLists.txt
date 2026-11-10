# lombscargle-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME lombscargle
    MODEL omp
    SOURCES main.cpp
    CATEGORIES signal
)
