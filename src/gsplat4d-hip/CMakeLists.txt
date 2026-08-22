# gsplat4d-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME gsplat4d
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
)
