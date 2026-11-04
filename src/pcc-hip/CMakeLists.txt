# pcc-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../pcc-cuda")

add_hecbench_benchmark(
    NAME pcc
    MODEL hip
    SOURCES device.cu ${INC_DIR}/main.cpp
    CATEGORIES bioinformatics
    LINK_LIBRARIES hipblas
)
