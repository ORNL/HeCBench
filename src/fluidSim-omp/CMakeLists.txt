# fluidSim-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME fluidSim
    MODEL omp
    SOURCES kernels.cpp main.cpp
    CATEGORIES simulation
    COMPILE_OPTIONS -DVERIFY
)
