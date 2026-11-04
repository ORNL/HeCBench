# face-omp/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../face-sycl")

add_hecbench_benchmark(
    NAME face
    MODEL omp
    SOURCES haar.cpp
    ${INC_DIR}/image.cpp
    ${INC_DIR}/main.cpp
    ${INC_DIR}/rectangles.cpp
    ${INC_DIR}/stdio-wrapper.cpp
    CATEGORIES imaging
)
