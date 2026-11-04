# mf-sgd-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME mf-sgd
    MODEL sycl
    SOURCES
        src/main.cpp
        src/sgd.cpp
    INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
    LINK_LIBRARIES -lpthread
    CATEGORIES ml
)
