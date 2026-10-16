# tgvnn-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME tgvnn
    MODEL cuda
    SOURCES
        src/main.cu
        src/ra.cu
    INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
    LINK_LIBRARIES cufft cublas cusolver
    CATEGORIES imaging algorithms
)
