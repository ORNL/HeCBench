# pointwise-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME pointwise
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
