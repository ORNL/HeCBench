# mcpr-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME mcpr
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
