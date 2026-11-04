# rowwiseMoments-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME rowwiseMoments
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
