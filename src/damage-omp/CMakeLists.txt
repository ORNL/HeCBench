# damage-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME damage
    MODEL omp
    SOURCES main.cpp
    CATEGORIES simulation
)
