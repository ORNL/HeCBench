# binomial-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME binomial
    MODEL sycl
    SOURCES kernel.cpp main.cpp reference.cpp
    CATEGORIES finance
)
