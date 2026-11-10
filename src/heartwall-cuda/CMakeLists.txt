# heartwall-cuda/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/util")

UNZIPFILE("${CMAKE_CURRENT_LIST_DIR}/../data/heartwall/heartwall.tar.bz")

add_hecbench_benchmark(
    NAME heartwall
    MODEL cuda
    SOURCES main.cu kernel/kernel.cu util/avi/avilib.c util/avi/avimod.c util/file/file.c
    CATEGORIES simulation
    INCLUDE_DIRS  ${INC_DIR}/avi ${INC_DIR}/file
)
