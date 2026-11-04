# sobel-omp/CMakeLists.txt

set(SDK_DIR "${CMAKE_CURRENT_LIST_DIR}/../include")

add_hecbench_benchmark(
    NAME sobel
    MODEL omp
    SOURCES main.cpp reference.cpp
    CATEGORIES imaging
    INCLUDE_DIRS ${SDK_DIR}
)
