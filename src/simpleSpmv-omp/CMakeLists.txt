# simpleSpmv-omp/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../simpleSpmv-cuda")

add_hecbench_benchmark(
    NAME simpleSpmv
    MODEL omp
    SOURCES kernels.cpp ${INC_DIR}/utils.cpp ${INC_DIR}/main.cpp  
    CATEGORIES math
)
