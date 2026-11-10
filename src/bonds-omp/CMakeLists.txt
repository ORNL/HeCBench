# bonds-omp/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../bonds-sycl")

add_hecbench_benchmark(
    NAME bonds
    MODEL omp
    SOURCES bondsEngine.cpp
    CATEGORIES finance
    INCLUDE_DIRS ${INC_DIR}
)
