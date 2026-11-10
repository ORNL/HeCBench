# unfold-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME unfold
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
