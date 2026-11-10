# rushlarsen-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME rushlarsen
    MODEL omp
    SOURCES main.cpp reference.cpp utils.cpp
    CATEGORIES simulation
)
