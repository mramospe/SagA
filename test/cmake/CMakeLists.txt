#
# Example of a CMake project depending on SagA
#
cmake_minimum_required(VERSION 2.8)
project(SagATest)
set(CMAKE_CXX_STANDARD 17)
find_package(SagA REQUIRED)
message("-- Found SagA version ${SagA_VERSION}")
message("-- Including directories ${SagA_INCLUDE_DIR}")
add_executable(main main.cpp)
target_include_directories(main PUBLIC ${SagA_INCLUDE_DIR})
