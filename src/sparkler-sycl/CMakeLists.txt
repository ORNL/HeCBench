# sparkler-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME sparkler
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
    LINK_LIBRARIES -qmkl=parallel
)
