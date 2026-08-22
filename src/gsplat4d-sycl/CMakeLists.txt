# gsplat4d-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME gsplat4d
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
)
