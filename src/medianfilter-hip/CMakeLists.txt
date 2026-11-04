# medianfilter-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../boxfilter-sycl")
set(INC_DIR2 "${CMAKE_CURRENT_LIST_DIR}/../medianfilter-cuda")

add_hecbench_benchmark(
    NAME medianfilter
    MODEL hip
    SOURCES main.cu ${INC_DIR}/shrUtils.cpp  ${INC_DIR2}/MedianFilterHost.cpp
    CATEGORIES imaging
    INCLUDE_DIRS ${INC_DIR}
)
