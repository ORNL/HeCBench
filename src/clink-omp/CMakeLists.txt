# clink-omp/CMakeLists.txt

UNZIPFILE("${CMAKE_CURRENT_LIST_DIR}/data.tar.gz")

add_hecbench_benchmark(
    NAME clink
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
