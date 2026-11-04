# morphology-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME morphology
    MODEL omp
    SOURCES main.cpp morphology.cpp
    CATEGORIES imaging
)
