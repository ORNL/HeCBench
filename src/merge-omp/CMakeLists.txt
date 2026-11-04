# merge-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME merge
    MODEL omp
    SOURCES main.cpp
    CATEGORIES sorting
)
