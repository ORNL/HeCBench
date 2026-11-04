# concat-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME concat
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
