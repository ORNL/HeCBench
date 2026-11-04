# projectile-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME projectile
    MODEL omp
    SOURCES Projectile.cpp
    CATEGORIES simulation
)
