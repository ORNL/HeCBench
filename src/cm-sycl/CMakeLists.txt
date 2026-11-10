# cm-sycl/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../cm-cuda")

add_hecbench_benchmark(
    NAME cm
    MODEL sycl
    SOURCES main.cpp process.cpp ${INC_DIR}/io.cpp
    CATEGORIES bioinformatics
)
