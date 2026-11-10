# slit-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME slit
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
    LINK_LIBRARIES -qmkl=parallel
)
