# cfd-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME cfd
    MODEL omp
    SOURCES euler3d.cpp
    CATEGORIES simulation
)
