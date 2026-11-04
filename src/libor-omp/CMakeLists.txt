# libor-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME libor
    MODEL omp
    SOURCES main.cpp
    CATEGORIES finance
)
