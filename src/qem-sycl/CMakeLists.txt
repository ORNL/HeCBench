# qem-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME qem
    MODEL sycl
    SOURCES gpu_solver.cpp main.cpp
    CATEGORIES simulation
)
