# kmeans-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME kmeans
    MODEL hip
    SOURCES cluster.cu getopt.cpp kmeans.cpp read_input.cpp rmse.cpp
    CATEGORIES ml
)
