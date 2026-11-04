# randomAccess-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME randomAccess
    MODEL omp
    SOURCES main.cpp
    CATEGORIES bandwidth
)
