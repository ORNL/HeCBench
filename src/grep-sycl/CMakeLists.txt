# grep-sycl/CMakeLists.txt

REQUIREDATADIR("${CMAKE_CURRENT_LIST_DIR}/../data/grep/testcases")

add_hecbench_benchmark(
    NAME grep
    MODEL sycl
    SOURCES main.cpp nfautil.cpp regex.cpp
    CATEGORIES search
)
