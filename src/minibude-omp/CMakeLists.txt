# minibude-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME minibude
    MODEL omp
    SOURCES kernel.cpp main.cpp
    CATEGORIES bioinformatics
)
