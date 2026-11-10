# scel-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME scel
    MODEL omp
    SOURCES main.cpp
    ENABLE_FASTMATH true
    CATEGORIES ml
)
