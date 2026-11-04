# pcc-sycl/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../pcc-cuda")

add_hecbench_benchmark(
    NAME pcc
    MODEL sycl
    SOURCES device.cpp ${INC_DIR}/main.cpp
    CATEGORIES bioinformatics
    LINK_LIBRARIES -qmkl=parallel
)
