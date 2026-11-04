# wlcpow-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME wlcpow
    MODEL omp
    SOURCES main.cpp
    CATEGORIES simulation
)
