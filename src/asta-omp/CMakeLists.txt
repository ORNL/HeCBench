# asta-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME asta
    MODEL omp
    SOURCES main.cpp
    CATEGORIES language
)
