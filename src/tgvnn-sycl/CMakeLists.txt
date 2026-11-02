# tgvnn-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME tgvnn
    MODEL sycl
    SOURCES
        src/main.cpp
        src/tgv.cpp
        src/ra.cpp
    INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
    LINK_LIBRARIES -qmkl=parallel
    CATEGORIES imaging algorithms
)
