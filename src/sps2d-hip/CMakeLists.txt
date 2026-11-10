# sps2d-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../spd2s-cuda")

add_hecbench_benchmark(
    NAME sps2d
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
    INCLUDE_DIRS ${INC_DIR}
    LINK_LIBRARIES hipsparse
)
