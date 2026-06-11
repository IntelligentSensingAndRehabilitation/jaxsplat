cmake_minimum_required(VERSION 3.24)

set(CMAKE_CUDA_FLAGS -std=c++17) # must be set before CUDA detection

# CUDA architectures. An explicit list passed via -DCMAKE_CUDA_ARCHITECTURES
# (e.g. from CMAKE_ARGS in a pip/scikit-build-core build) always wins. When the
# caller passes nothing we default to a fixed real-architecture list rather than
# `native`: a `git+`/sdist build commonly runs with no GPU visible, where
# `native` silently falls back to nvcc's lowest default arch and produces a
# wheel that won't run on the target card. The default below covers Ampere
# (80/86), Ada (89) and Hopper (90); override it for other GPUs. Must be decided
# before project() enables the CUDA language.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 80;86;89;90)
endif()

project(_jaxsplat LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

set(PYBIND11_NEWPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

include_directories(${CMAKE_CURRENT_LIST_DIR}/lib)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

pybind11_add_module(
    _jaxsplat
    ${CMAKE_CURRENT_LIST_DIR}/lib/ffi.cu
    ${CMAKE_CURRENT_LIST_DIR}/lib/ops.cu
    ${CMAKE_CURRENT_LIST_DIR}/lib/kernels/forward.cu
    ${CMAKE_CURRENT_LIST_DIR}/lib/kernels/backward.cu
    ${CMAKE_CURRENT_LIST_DIR}/lib/kernels/other.cu
)

if(DEFINED SKBUILD)
    set(JAXSPLAT_INSTALL_DEST ${SKBUILD_PLATLIB_DIR})
else()
    set(JAXSPLAT_INSTALL_DEST ${CMAKE_CURRENT_LIST_DIR})
endif()

message(STATUS "Installing _jaxsplat to ${JAXSPLAT_INSTALL_DEST}")
install(TARGETS _jaxsplat LIBRARY DESTINATION ${JAXSPLAT_INSTALL_DEST})
