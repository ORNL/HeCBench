# nlll-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME nlll
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
