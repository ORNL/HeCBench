# groupnorm-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME groupnorm
    MODEL omp
    SOURCES main.cpp
    CATEGORIES imaging
)
