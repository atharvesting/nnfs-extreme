# CMakeList.txt : CMake project for NNFS_Extreme, include source and define
# project specific logic here.
#
# Add source to this project's executable.
add_executable (NNFS_Extreme
    "NNFS_Extreme.cpp"
    "NN.cpp"
    "data_loaders.cpp"
    "activation_functions.cpp"
)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET NNFS_Extreme PROPERTY CXX_STANDARD 20)
endif()

target_link_libraries(NNFS_Extreme PRIVATE Spalten)

