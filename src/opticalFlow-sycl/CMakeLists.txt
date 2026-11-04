# opticalFlow-sycl/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../opticalFlow-cuda")

add_hecbench_benchmark(
    NAME opticalFlow
    MODEL sycl
    SOURCES flowSYCL.cpp flowGold.cpp main.cpp
    CATEGORIES imaging
    INCLUDE_DIRS ${INC_DIR}/Common
)
