# gsplat4d-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME gsplat4d
    MODEL omp
    SOURCES main.cpp
    CATEGORIES simulation
)
