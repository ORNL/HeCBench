# su3-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME su3
    MODEL sycl
    SOURCES su3_nn_bench.cpp
    CATEGORIES simulation
    COMPILE_OPTIONS -DMILC_COMPLEX
)
