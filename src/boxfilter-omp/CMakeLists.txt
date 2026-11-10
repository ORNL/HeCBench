# boxfilter-omp/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../boxfilter-sycl")

add_hecbench_benchmark(
    NAME boxfilter
    MODEL omp
    SOURCES main.cpp ${INC_DIR}/shrUtils.cpp ${INC_DIR}/reference.cpp ${INC_DIR}/cmd_arg_reader.cpp
    CATEGORIES imaging
    INCLUDE_DIRS ${INC_DIR}
)
