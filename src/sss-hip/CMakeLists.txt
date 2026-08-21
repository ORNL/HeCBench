# sss-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME sss
    MODEL hip
    SOURCES main.cu
    INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}/../sss-cuda
    CATEGORIES algorithms
)
