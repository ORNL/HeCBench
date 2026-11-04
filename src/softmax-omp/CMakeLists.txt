# softmax-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME softmax
    MODEL omp
    SOURCES main.cpp
    ENABLE_FASTMATH true
    CATEGORIES ml
)
