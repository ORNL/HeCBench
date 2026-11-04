# aidw-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME aidw
    MODEL omp
    SOURCES main.cpp
    ENABLE_FASTMATH true
    CATEGORIES geoscience
)
