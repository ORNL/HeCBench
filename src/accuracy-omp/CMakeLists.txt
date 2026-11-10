# accuracy-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME accuracy
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
