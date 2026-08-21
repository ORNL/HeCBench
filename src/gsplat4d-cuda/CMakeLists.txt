# gsplat4d-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME gsplat4d
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
)
