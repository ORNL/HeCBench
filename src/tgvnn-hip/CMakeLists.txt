# tgvnn-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME tgvnn
    MODEL hip
    SOURCES
        src/main.cu
        src/ra.cu
    INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
    LINK_LIBRARIES hipfft hipblas hipsolver
    CATEGORIES imaging algorithms
)
