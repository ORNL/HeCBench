# distort-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME distort
    MODEL omp
    SOURCES distort.cpp main.cpp
    CATEGORIES imaging
)
