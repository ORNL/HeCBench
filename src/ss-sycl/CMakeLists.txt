# ss-sycl/CMakeLists.txt

REQUIREDATAFILE("${CMAKE_CURRENT_LIST_DIR}/../data/ss/StringSearch_Input.txt")

add_hecbench_benchmark(
    NAME ss
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES search
)
