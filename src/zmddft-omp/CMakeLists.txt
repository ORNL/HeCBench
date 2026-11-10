# zmddft-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME zmddft
    MODEL omp
    SOURCES main.cpp
    CATEGORIES signal
)
