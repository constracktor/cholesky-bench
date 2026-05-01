cmake_minimum_required(VERSION 3.14)
# Top-level coordinator for *source formatting only*. Each subdirectory owns its
# own standalone CMake project (with its own dependencies and its own
# compile.sh); this file exists so the clang-format / cmake-format integration
# can be configured once for the whole repository.
project(cholesky_bench LANGUAGES NONE)

include(FetchContent)
FetchContent_Declare(
  format
  GIT_REPOSITORY https://github.com/TheLartians/Format.cmake.git
  GIT_TAG v1.8.1
  QUIET)
FetchContent_MakeAvailable(format)
