# gmm-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME gmm
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
