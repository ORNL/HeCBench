# sosfil-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME sosfil
    MODEL omp
    SOURCES main.cpp
    CATEGORIES signal
)
