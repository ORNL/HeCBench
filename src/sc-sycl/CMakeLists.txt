# sc-sycl/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../sc-cuda")

add_hecbench_benchmark(
    NAME sc
    MODEL sycl
    SOURCES device_sc.cpp ${INC_DIR}/host_sc.cpp main.cpp
    CATEGORIES compression
    LINK_LIBRARIES pthread
)
