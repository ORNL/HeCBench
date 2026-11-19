# ss-cuda/CMakeLists.txt

REQUIREDATAFILE("${CMAKE_CURRENT_LIST_DIR}/../data/ss/StringSearch_Input.txt")

add_hecbench_benchmark(
    NAME ss
    MODEL cuda
    SOURCES main.cu
    CATEGORIES search
)
