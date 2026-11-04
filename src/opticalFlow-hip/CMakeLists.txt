# opticalFlow-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../opticalFlow-cuda/Common")

add_hecbench_benchmark(
    NAME opticalFlow
    MODEL hip
    SOURCES flowGold.cu flowHIP.cu main.cu
    INCLUDE_DIRS ${INC_DIR}
    CATEGORIES imaging
)
