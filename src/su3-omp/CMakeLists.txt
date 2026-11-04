# su3-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME su3
    MODEL omp
    SOURCES su3_nn_bench.cpp
    CATEGORIES simulation
)
