# projectile-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME projectile
    MODEL sycl
    SOURCES Projectile.cpp
    CATEGORIES simulation
)
