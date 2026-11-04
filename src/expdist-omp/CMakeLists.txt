# expdist-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME expdist
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
