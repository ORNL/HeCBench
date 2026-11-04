# complex-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME complex
    MODEL omp
    SOURCES main.cpp
    CATEGORIES math
)
