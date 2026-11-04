# sheath-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME sheath
    MODEL omp
    SOURCES main.cpp
    CATEGORIES simulation
)
