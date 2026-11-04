# clenergy-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME clenergy
    MODEL omp
    SOURCES WKFUtils.cpp clenergy.cpp
    CATEGORIES simulation
)
