# logprob-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME logprob
    MODEL omp
    SOURCES main.cpp
    ENABLE_FASTMATH true
    CATEGORIES ml
)
