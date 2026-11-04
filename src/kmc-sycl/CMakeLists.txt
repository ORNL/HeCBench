# kmc-sycl/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../kmc-cuda")

add_hecbench_benchmark(
    NAME kmc
    MODEL sycl
    SOURCES main.cpp ${INC_DIR}/svm.cpp
    INCLUDE_DIRS ${INC_DIR}
    CATEGORIES ml
    LINK_LIBRARIES -qmkl=parallel
)
