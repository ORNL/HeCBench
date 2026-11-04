# background-subtract-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME background-subtract
    MODEL omp
    SOURCES main.cpp
    CATEGORIES imaging
)
