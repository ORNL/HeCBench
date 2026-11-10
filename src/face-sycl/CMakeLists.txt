# face-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME face
    MODEL sycl
    SOURCES haar.cpp image.cpp main.cpp rectangles.cpp stdio-wrapper.cpp
    CATEGORIES imaging
)
