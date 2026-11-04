# car-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME car
    MODEL omp
    SOURCES main.cpp
    CATEGORIES imaging
)
