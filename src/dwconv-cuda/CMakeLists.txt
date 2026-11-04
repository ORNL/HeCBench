# dwconv-cuda/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../tensorAccessor-cuda")

add_hecbench_benchmark(
    NAME dwconv
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
    INCLUDE_DIRS ${INC_DIR}
)
