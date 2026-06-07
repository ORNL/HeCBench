# opticalFlow-omp/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../opticalFlow-cuda/Common")

add_hecbench_benchmark(
    NAME opticalFlow
    MODEL omp
    SOURCES flowGold.cpp flowOMP.cpp main.cpp
    INCLUDE_DIRS ${INC_DIR}
    CATEGORIES algorithms
)
