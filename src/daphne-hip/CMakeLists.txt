# daphne-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME daphne
    MODEL hip
    SOURCES
        ${CMAKE_SOURCE_DIR}/src/daphne-cuda/src/common/main.cpp
        src/points2image/kernel.cu
    INCLUDE_DIRS
        ${CMAKE_SOURCE_DIR}/src/daphne-cuda/src/include
        ${CMAKE_SOURCE_DIR}/src/daphne-cuda/src/points2image
    CATEGORIES automotive
)
