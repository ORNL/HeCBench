# vanGenuchten-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME vanGenuchten
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
)
