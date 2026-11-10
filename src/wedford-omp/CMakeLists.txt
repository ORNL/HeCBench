# wedford-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME wedford
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
