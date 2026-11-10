# extend2-omp/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../extend2-sycl")

add_hecbench_benchmark(
    NAME extend2
    MODEL omp
    SOURCES main.cpp ${INC_DIR}/read_data.cpp
    CATEGORIES bioinformatics
    INCLUDE_DIRS ${INC_DIR}
)
