# daphne-sycl/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/src/include")
set(SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/../daphne-cuda/src/points2image")

add_hecbench_benchmark(
    NAME daphne
    MODEL sycl
    SOURCES
        src/common/main.cpp
        src/points2image/kernel.cpp
    INCLUDE_DIRS ${INC_DIR} ${SRC_DIR}
    CATEGORIES automotive
)
