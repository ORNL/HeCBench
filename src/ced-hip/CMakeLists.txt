# ced-hip/CMakeLists.txt

REQUIREDATADIR("${CMAKE_CURRENT_LIST_DIR}/../data/ced/input")
REQUIREDATADIR("${CMAKE_CURRENT_LIST_DIR}/../data/ced/output")

add_hecbench_benchmark(
    NAME ced
    MODEL hip
    SOURCES kernel_reference.cu main.cu
    CATEGORIES imaging
    LINK_LIBRARIES pthread
)
