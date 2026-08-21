# bicgstab-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME bicgstab
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
    LINK_LIBRARIES -qmkl=parallel
)
