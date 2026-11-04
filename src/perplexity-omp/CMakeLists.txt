# perplexity-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME perplexity
    MODEL omp
    SOURCES main.cpp
    ENABLE_FASTMATH true
    CATEGORIES ml
)
