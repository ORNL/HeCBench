# d2q9-bgk-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME d2q9-bgk
    MODEL omp
    SOURCES main.cpp
    CATEGORIES simulation
)
