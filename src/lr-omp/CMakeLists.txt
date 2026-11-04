# lr-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME lr
    MODEL omp
    SOURCES linear_iter.cpp linear_par.cpp main.cpp
    CATEGORIES ml
)
