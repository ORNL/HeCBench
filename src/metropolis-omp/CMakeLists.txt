# metropolis-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME metropolis
    MODEL omp
    SOURCES heap.cpp main.cpp utils.cpp
    CATEGORIES simulation
)
