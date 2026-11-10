# gc-omp/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../mis-cuda")

add_hecbench_benchmark(
    NAME gc
    MODEL omp
    SOURCES main.cpp
    INCLUDE_DIRS ${INC_DIR}
    CATEGORIES graph
)
