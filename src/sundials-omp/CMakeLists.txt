# sundials-omp/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../sundials-cuda")

add_hecbench_benchmark(
    NAME sundials
    MODEL omp
    SOURCES kernels.cpp ${INC_DIR}/utils.cpp ${INC_DIR}/main.cpp
    CATEGORIES simulation algorithms math
)
