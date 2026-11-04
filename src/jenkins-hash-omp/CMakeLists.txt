# jenkins-hash-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME jenkins-hash
    MODEL omp
    SOURCES main.cpp
    CATEGORIES encoding
)
