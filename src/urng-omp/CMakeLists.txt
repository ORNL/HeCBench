# urng-omp/CMakeLists.txt

REQUIREDATAFILE("${CMAKE_CURRENT_LIST_DIR}/../data/urng/URNG_Input.bmp")

set(SDK_DIR "${CMAKE_CURRENT_LIST_DIR}/../include")

add_hecbench_benchmark(
    NAME urng
    MODEL omp
    SOURCES main.cpp
    CATEGORIES rng
    INCLUDE_DIRS ${SDK_DIR}
)
