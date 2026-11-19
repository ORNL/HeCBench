# grep-hip/CMakeLists.txt

REQUIREDATADIR("${CMAKE_CURRENT_LIST_DIR}/../data/grep/testcases")

add_hecbench_benchmark(
    NAME grep
    MODEL hip
    SOURCES main.cu nfautil.cu regex.cu
    CATEGORIES search
)
