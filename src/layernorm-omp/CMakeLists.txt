# layernorm-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME layernorm
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
