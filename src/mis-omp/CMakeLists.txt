# mis-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME mis
    MODEL omp
    SOURCES main.cpp
    CATEGORIES graph
)
