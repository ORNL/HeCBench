# present-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME present
    MODEL omp
    SOURCES main.cpp
    CATEGORIES cryptography
)
