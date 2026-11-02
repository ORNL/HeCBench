# rmsnorm-rope-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME rmsnorm-rope
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
