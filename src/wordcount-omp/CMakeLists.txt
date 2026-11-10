# wordcount-omp/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../wordcount-sycl")

add_hecbench_benchmark(
    NAME wordcount
    MODEL omp
    SOURCES wc.cpp ${INC_DIR}/main.cpp
    INCLUDE_DIRS ${INC_DIR}
    CATEGORIES language
)
