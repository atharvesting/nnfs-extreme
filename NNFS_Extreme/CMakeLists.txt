# CMakeList.txt : CMake project for NNFS_Extreme, include source and define
# project specific logic here.
#

# ── Shared core (compiled once, linked by both targets) ─────────────────────
add_library(nnfs_core STATIC
    "NN.cpp"
    "data_loaders.cpp"
    "activation_functions.cpp"
    "utils.cpp"
)
set_property(TARGET nnfs_core PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET nnfs_core PROPERTY CXX_STANDARD 20)
set_property(TARGET nnfs_core PROPERTY CXX_SCAN_FOR_MODULES OFF)
target_include_directories(nnfs_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nnfs_core PUBLIC Spalten)

# ── Executable target ────────────────────────────────────────────────────────
add_executable(NNFS_Extreme
    "NNFS_Extreme.cpp"
    "utils.hpp"
    "examples/model_export_n_import.hpp"
    examples/import_model.hpp
)
set_property(TARGET NNFS_Extreme PROPERTY CXX_STANDARD 20)
target_link_libraries(NNFS_Extreme PRIVATE nnfs_core)

# ── Python extension module ──────────────────────────────────────────────────
pybind11_add_module(nnfs_extreme "python_bindings.cpp")
set_property(TARGET nnfs_extreme PROPERTY CXX_STANDARD 20)
set_property(TARGET nnfs_extreme PROPERTY CXX_SCAN_FOR_MODULES OFF)
target_link_libraries(nnfs_extreme PRIVATE nnfs_core)

# Output the .pyd directly into MNIST_CV/ at link time
set_target_properties(nnfs_extreme PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/MNIST_CV"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/MNIST_CV"
)

# Build nnfs_extreme whenever NNFS_Extreme is built
add_dependencies(NNFS_Extreme nnfs_extreme)

