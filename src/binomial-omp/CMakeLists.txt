# binomial-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME binomial
    MODEL omp
    SOURCES kernel.cpp main.cpp reference.cpp
    CATEGORIES finance
)
