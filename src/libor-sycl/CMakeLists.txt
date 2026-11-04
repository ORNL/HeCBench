# libor-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME libor
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES finance
)
