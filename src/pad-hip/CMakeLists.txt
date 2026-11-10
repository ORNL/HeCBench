# pad-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../pad-cuda")

add_hecbench_benchmark(
    NAME pad
    MODEL hip
    SOURCES main.cu kernel.cu ${INC_DIR}/host_task.cpp
    CATEGORIES language
    LINK_LIBRARIES pthread
)
