# mt-hip/CMakeLists.txt

FETCHDATA("${CMAKE_CURRENT_LIST_DIR}/../data/mt/fetch.sh")

add_hecbench_benchmark(
    NAME mt
    MODEL hip
    SOURCES MT.cu MT_gold.cu genmtrand.cu
    CATEGORIES rng
)
