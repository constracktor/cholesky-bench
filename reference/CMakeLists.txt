cmake_minimum_required(VERSION 3.23)
project(cholesky_reference)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# What to build?
option(BUILD_CORE "Build the core library" ON)
option(ENABLE_MKL "Enable Intel oneMKL support (threaded)" OFF)
option(
  ENABLE_VALIDATION
  "Compute ||A - L*L^T||_F / ||A||_F after each factorization (off by default)"
  OFF)
option(ENABLE_FORMAT_TARGETS "Enable clang-format / cmake-format targets"
       ${PROJECT_IS_TOP_LEVEL})

if(ENABLE_FORMAT_TARGETS)
  find_package(format QUIET)
  if(NOT format_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      format
      GIT_REPOSITORY https://github.com/TheLartians/Format.cmake.git
      GIT_TAG v1.8.1
      QUIET)
    FetchContent_MakeAvailable(format)
  endif()
endif()

if(NOT CMAKE_SKIP_INSTALL_RULES)
  # Our installs follow the standard GNU directory layout. This include needs to
  # come first since we need the CMAKE_INSTALL_* in the CMakeLists.txt of each
  # target.
  include(GNUInstallDirs)
endif()

if(BUILD_CORE)
  if(ENABLE_MKL)
    # Threaded Intel oneMKL: ask MKL to use its OpenMP runtime ('intel_thread').
    # This is the only difference from the OpenMP/HPX builds, which pin
    # MKL_THREADING=sequential because they parallelise at the tile level.
    # Here the parallelism lives inside dpotrf itself, so we want the
    # vendor-threaded backend.
    set(MKL_INTERFACE_FULL "intel_lp64")
    set(MKL_THREADING "intel_thread")
    find_package(MKL CONFIG REQUIRED)

    if(MKL_FOUND)
      message(STATUS "Intel oneMKL Library found (threaded: ${MKL_THREADING})")
    else()
      message(FATAL_ERROR "No BLAS Library found")
    endif()
  else()
    # Threaded OpenBLAS. The library name is the same as the sequential one,
    # but the Spack environment loaded by compile.sh selects an OpenBLAS built
    # with threads=openmp.
    find_library(OpenBLAS_LIB NAMES openblas REQUIRED)

    if(OpenBLAS_LIB)
      message(STATUS "OpenBLAS Library found at ${OpenBLAS_LIB}")
      find_path(
        OpenBLAS_INCLUDE_DIR
        NAMES cblas.h
        PATH_SUFFIXES openblas)
      if(NOT OpenBLAS_INCLUDE_DIR)
        message(FATAL_ERROR "OpenBLAS include directory not found")
      endif()

      message(STATUS "OpenBLAS include dir: ${OpenBLAS_INCLUDE_DIR}")
    else()
      message(FATAL_ERROR "No BLAS Library found")
    endif()
  endif()

  # OpenMP is required for the matrix-generation parallel loop and to pick up
  # the OpenMP runtime that threaded OpenBLAS / threaded MKL share.
  find_package(OpenMP REQUIRED)

  add_subdirectory(core)

  # Add the executable
  add_executable(cholesky_reference main.cpp)

  # Link the libraries
  target_link_libraries(cholesky_reference PUBLIC Cholesky::core
                                                  OpenMP::OpenMP_CXX)

  if(ENABLE_VALIDATION)
    target_compile_definitions(cholesky_reference PRIVATE ENABLE_VALIDATION)
  endif()
endif()
