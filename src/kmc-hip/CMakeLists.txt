# kmc-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../kmc-cuda")

add_hecbench_benchmark(
    NAME kmc
    MODEL hip
    SOURCES main.cu ${INC_DIR}/svm.cpp
    CATEGORIES ml
    LINK_LIBRARIES hipblas
)
