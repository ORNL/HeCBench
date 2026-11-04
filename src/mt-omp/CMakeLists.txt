# mt-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME mt
    MODEL omp
    SOURCES MT.cpp MT_gold.cpp genmtrand.cpp
    CATEGORIES rng
)
