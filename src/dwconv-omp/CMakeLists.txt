# dwconv-omp/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../dwconv-cuda")

add_hecbench_benchmark(
    NAME dwconv
    MODEL omp
    SOURCES main.cpp
    CATEGORIES algorithms
    INCLUDE_DIRS ${INC_DIR}
)
