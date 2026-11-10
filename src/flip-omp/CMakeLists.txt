# flip-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME flip
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
