cmake_minimum_required(VERSION 3.15)
project(pracpy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release build = -O3 -DNDEBUG by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Ask the build-env Python for NumPy's include dir
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
    "import numpy, sys; sys.stdout.write(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    RESULT_VARIABLE NUMPY_RESULT
)
if(NOT NUMPY_RESULT EQUAL 0 OR NUMPY_INCLUDE_DIR STREQUAL "")
    # Fallback: let the user pass -DNUMPY_INCLUDE_DIR=...
    if(NOT NUMPY_INCLUDE_DIR)
        message(FATAL_ERROR
            "NumPy not found in the build environment.\n"
            "Pass -DNUMPY_INCLUDE_DIR=\$(python -c 'import numpy; print(numpy.get_include())')")
    endif()
endif()
message(STATUS "NumPy include dir: ${NUMPY_INCLUDE_DIR}")

pybind11_add_module(_core MODULE src/pracpy/_core.cpp)

target_include_directories(_core PRIVATE ${NUMPY_INCLUDE_DIR})

# Optimisation flags
target_compile_options(_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -march=native -ffast-math>
    $<$<CXX_COMPILER_ID:MSVC>:/O2>
)

install(TARGETS _core DESTINATION pracpy)
