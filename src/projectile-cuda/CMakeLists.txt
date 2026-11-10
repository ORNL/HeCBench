# projectile-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME projectile
    MODEL cuda
    SOURCES Projectile.cu
    CATEGORIES simulation
)
