# ss-hip/CMakeLists.txt

REQUIREDATAFILE("${CMAKE_CURRENT_LIST_DIR}/../data/ss/StringSearch_Input.txt")

add_hecbench_benchmark(
    NAME ss
    MODEL hip
    SOURCES main.cu
    CATEGORIES search
)
