# s8n-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME s8n
    MODEL omp
    SOURCES main.cpp
    CATEGORIES search
)
