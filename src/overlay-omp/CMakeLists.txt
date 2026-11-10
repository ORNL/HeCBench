# overlay-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME overlay
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
