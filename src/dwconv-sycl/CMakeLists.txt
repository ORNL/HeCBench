# dwconv-sycl/CMakeLists.txt

set(INC_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../tensorAccessor-cuda"
    "${CMAKE_CURRENT_LIST_DIR}/../dwconv-cuda"
)

add_hecbench_benchmark(
    NAME dwconv
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
    INCLUDE_DIRS ${INC_DIR}
)
