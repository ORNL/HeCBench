# mf-sgd-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME mf-sgd
    MODEL cuda
    SOURCES
        src/main.cpp
        src/sgd.cu
    INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
    LINK_LIBRARIES -lpthread
    CATEGORIES ml
)
