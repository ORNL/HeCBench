# wordcount-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../wordcount-sycl")

add_hecbench_benchmark(
    NAME wordcount
    MODEL hip
    SOURCES ${INC_DIR}/main.cpp wc.cu
    CATEGORIES language
)
