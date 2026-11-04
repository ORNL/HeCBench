# grep-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME grep
    MODEL omp
    SOURCES main.cpp nfautil.cpp pnfa.cpp putil.cpp regex.cpp
    CATEGORIES search
)
