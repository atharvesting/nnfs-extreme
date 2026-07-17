# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.14)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("NNFS_Extreme")

# ============================================================================
# GLOBAL MAXIMUM OPTIMIZATIONS & VECTORIZATION CONFIGURATION
# ============================================================================
if(MSVC)
    # Append high-performance execution flags to global CXX flags for Release modes
    string(APPEND CMAKE_CXX_FLAGS_RELEASE " /O2 /Oi /Ot /Oy /GL /arch:AVX2 /fp:fast")
    string(APPEND CMAKE_CXX_FLAGS_RELWITHDEBINFO " /O2 /Oi /Ot /Oy /GL /arch:AVX2 /fp:fast")
    
    # Enable Link-Time Code Generation globally for optimized configurations
    string(APPEND CMAKE_EXE_LINKER_FLAGS_RELEASE " /LTCG")
    string(APPEND CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO " /LTCG")
    
    # Optional diagnostic tool: Uncomment the line below to see exactly which 
    # loops Visual Studio successfully vectorizes in the build output.
    # string(APPEND CMAKE_CXX_FLAGS " /Qvec-report:2")
else()
    # Fallback flags maximizing optimization/vectorization for GCC/Clang
    string(APPEND CMAKE_CXX_FLAGS_RELEASE " -O3 -march=native -ffast-math")
    string(APPEND CMAKE_CXX_FLAGS_RELWITHDEBINFO " -O3 -march=native -ffast-math")
endif()
# ============================================================================

add_subdirectory("third_party/Spalten")

add_subdirectory ("NNFS_Extreme")
