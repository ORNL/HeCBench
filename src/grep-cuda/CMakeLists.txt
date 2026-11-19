# grep-cuda/CMakeLists.txt

REQUIREDATADIR("${CMAKE_CURRENT_LIST_DIR}/../data/grep/testcases")

add_hecbench_benchmark(
    NAME grep
    MODEL cuda
    SOURCES main.cu nfautil.cu regex.cu
    CATEGORIES search
)
