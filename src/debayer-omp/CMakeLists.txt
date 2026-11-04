# debayer-omp/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../debayer-sycl")

add_hecbench_benchmark(
    NAME debayer
    MODEL omp
    SOURCES main.cpp
    CATEGORIES imaging
    INCLUDE_DIRS ${INC_DIR}
)
