# perplexity-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME perplexity
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
