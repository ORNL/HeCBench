# medianfilter-sycl/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../boxfilter-sycl")
set(INC2_DIR "${CMAKE_CURRENT_LIST_DIR}/../medianfilter-cuda")

add_hecbench_benchmark(
    NAME medianfilter
    MODEL sycl
    SOURCES ${INC2_DIR}/MedianFilterHost.cpp main.cpp ${INC_DIR}/shrUtils.cpp
    CATEGORIES imaging
    INCLUDE_DIRS ${INC_DIR} ${INC2_DIR}
)
