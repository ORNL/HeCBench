# oziMMU-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME oziMMU
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
    LINK_LIBRARIES -qmkl=parallel
)
