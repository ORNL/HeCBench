# extend2-cuda/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../extend2-sycl")

add_hecbench_benchmark(
    NAME extend2
    MODEL cuda
    SOURCES main.cu ${INC_DIR}/read_data.cpp
    CATEGORIES bioinformatics
    INCLUDE_DIRS ${INC_DIR}
)
