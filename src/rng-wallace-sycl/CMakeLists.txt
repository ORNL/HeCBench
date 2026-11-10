# rng-wallace-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME rng-wallace
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES rng
)
