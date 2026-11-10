# lebesgue-omp/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../lebesgue-cuda")

add_hecbench_benchmark(
    NAME lebesgue
    MODEL omp
    SOURCES kernels.cpp
            ${INC_DIR}/main.cpp
            ${INC_DIR}/utils.cpp
    CATEGORIES math
)
