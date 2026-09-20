# swiglu-oai-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME swiglu-oai
    MODEL omp
    SOURCES main.cpp
    CATEGORIES algorithms
)
