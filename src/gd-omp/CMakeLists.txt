# gd-omp/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../gd-cuda")

add_hecbench_benchmark(
    NAME gd
    MODEL omp
    SOURCES main.cpp ${INC_DIR}/utils.cpp
    CATEGORIES ml
    INCLUDE_DIRS ${INC_DIR}
)
