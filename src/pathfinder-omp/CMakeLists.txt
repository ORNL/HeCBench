# pathfinder-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME pathfinder
    MODEL omp
    SOURCES main.cpp
    CATEGORIES simulation
)
