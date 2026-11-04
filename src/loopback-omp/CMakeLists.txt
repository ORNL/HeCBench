# loopback-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME loopback
    MODEL omp
    SOURCES main.cpp
    CATEGORIES simulation
)
