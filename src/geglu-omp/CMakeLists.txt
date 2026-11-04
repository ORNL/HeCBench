# geglu-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME geglu
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
