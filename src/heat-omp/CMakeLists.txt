# heat-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME heat
    MODEL omp
    SOURCES heat.cpp
    CATEGORIES simulation
)
