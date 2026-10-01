# schur-complement-omp/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../schur-complement-cuda")

add_hecbench_benchmark(
    NAME schur-complement
    MODEL omp
    SOURCES main.cpp
    CATEGORIES math
    INCLUDE_DIRS ${INC_DIR}
)
