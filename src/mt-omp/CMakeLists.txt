# mt-omp/CMakeLists.txt

FETCHDATA("${CMAKE_CURRENT_LIST_DIR}/../data/mt/fetch.sh")

add_hecbench_benchmark(
    NAME mt
    MODEL omp
    SOURCES MT.cpp MT_gold.cpp genmtrand.cpp
    CATEGORIES rng
)
