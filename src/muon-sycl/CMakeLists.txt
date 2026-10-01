# muon-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME muon
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
    LINK_LIBRARIES -qmkl=parallel
)
