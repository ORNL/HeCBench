# logic-rewrite-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME logic-rewrite
    MODEL sycl
    SOURCES aig_manager.cpp
            balance.cpp
            print.cpp
            refactor_core.cpp
            refactor.cpp
            refactor_mffc.cpp
            rewrite.cpp
            strash.cpp
            truth.cpp
            command_manager.cpp
            main.cpp
    CATEGORIES algorithms
)
