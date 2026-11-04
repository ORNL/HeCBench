# babelstream-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME babelstream
    MODEL omp
    SOURCES main.cpp
    CATEGORIES bandwidth
)
