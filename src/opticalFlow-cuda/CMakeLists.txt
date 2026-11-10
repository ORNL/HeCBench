# opticalFlow-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME opticalFlow
    MODEL cuda
    SOURCES flowCUDA.cu flowGold.cu main.cu
    CATEGORIES imaging
    INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/Common
)
