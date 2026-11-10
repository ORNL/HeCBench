# vanGenuchten-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME vanGenuchten
    MODEL omp
    SOURCES main.cpp
    CATEGORIES simulation
)
