# hwt1d-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME hwt1d
    MODEL omp
    SOURCES main.cpp reference.cpp
    CATEGORIES simulation
)
