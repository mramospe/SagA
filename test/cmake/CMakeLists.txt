#
# Example of a CMake project depending on Sag-A
#
cmake_minimum_required(VERSION 2.8)
project(Sag-ATest)
set(CMAKE_CXX_STANDARD 17)
find_package(Sag-A REQUIRED)
message("-- Found Sag-A version ${Sag-A_VERSION}")
message("-- Including directories ${Sag-A_INCLUDE_DIR}")
add_executable(main main.cpp)
target_include_directories(main PUBLIC ${Sag-A_INCLUDE_DIR})
