# bwt-omp/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../bwt-cuda")

add_hecbench_benchmark(
    NAME bwt
    MODEL omp
    SOURCES bwt.cpp ${INC_DIR}/main.cpp
    CATEGORIES compression
)
