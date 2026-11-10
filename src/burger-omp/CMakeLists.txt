# burger-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME burger
    MODEL omp
    SOURCES main.cpp
    CATEGORIES simulation
)
