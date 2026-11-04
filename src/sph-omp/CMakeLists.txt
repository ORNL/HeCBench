# sph-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME sph
    MODEL omp
    SOURCES fileio.cpp fluid.cpp geometry.cpp
    CATEGORIES simulation
)
