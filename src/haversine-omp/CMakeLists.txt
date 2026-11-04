# haversine-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME haversine
    MODEL omp
    SOURCES main.cpp distance.cpp
    ENABLE_FASTMATH true
    CATEGORIES geoscience
)
