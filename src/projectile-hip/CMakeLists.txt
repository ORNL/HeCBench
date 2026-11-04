# projectile-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME projectile
    MODEL hip
    SOURCES Projectile.cu
    CATEGORIES simulation
)
