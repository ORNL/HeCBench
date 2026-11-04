# multimaterial-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME multimaterial
    MODEL sycl
    SOURCES compact.cpp full_matrix.cpp multimat.cpp
    CATEGORIES simulation
)
