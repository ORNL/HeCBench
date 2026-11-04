# gamma-correction-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME gamma-correction
    MODEL omp
    SOURCES main.cpp
    CATEGORIES imaging
)
