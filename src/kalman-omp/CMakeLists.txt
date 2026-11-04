# kalman-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME kalman
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
