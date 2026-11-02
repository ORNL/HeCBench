# kmeans-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME kmeans
    MODEL sycl
    SOURCES cluster.c getopt.c kmeans.c read_input.c rmse.c
    CATEGORIES algorithms
    C_AS_CXX
)
