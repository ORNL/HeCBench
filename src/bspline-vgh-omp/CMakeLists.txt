# bspline-vgh-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME bspline-vgh
    MODEL omp
    SOURCES main.cpp
    CATEGORIES simulation
)
