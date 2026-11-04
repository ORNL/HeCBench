# cm-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../cm-cuda")

add_hecbench_benchmark(
    NAME cm
    MODEL hip
    SOURCES ${INC_DIR}/main.cpp process.cu ${INC_DIR}/io.cpp
    CATEGORIES bioinformatics
)
