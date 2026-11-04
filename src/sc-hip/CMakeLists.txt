# sc-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../sc-cuda")

add_hecbench_benchmark(
    NAME sc
    MODEL hip
    SOURCES device_sc.cu main.cu ${INC_DIR}/host_sc.cpp
    CATEGORIES compression
    LINK_LIBRARIES pthread
)
