cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0048 NEW)

option(SAGA_CUDA_ENABLED "Whether to enable CUDA or not (only used for tests and examples)" OFF)
option(INSTALL_TESTS "Whether to install the test scripts or not" OFF)
option(INSTALL_EXAMPLES "Whether to install the examples or not" OFF)

#
# If CUDA is enabled add it to the compile definitions so it propagates through the code
#
if(SAGA_CUDA_ENABLED)
   add_compile_definitions(SAGA_CUDA_ENABLED=true)
   set(PROJECT_LANGUAGES CXX CUDA)
   set(CMAKE_CUDA_STANDARD 17)
   set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
else(SAGA_CUDA_ENABLED)
   set(PROJECT_LANGUAGES CXX)
endif(SAGA_CUDA_ENABLED)

project("SagA" VERSION 0.1.0 DESCRIPTION "A header only c++ library to study point-to-point interactions"
LANGUAGES ${PROJECT_LANGUAGES})

set(CMAKE_CXX_STANDARD 17)

add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(
  ${PROJECT_NAME}
  INTERFACE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
$<INSTALL_INTERFACE:include>)

include(CMakePackageConfigHelpers)
write_basic_package_version_file("${PROJECT_BINARY_DIR}/cmake/${PROJECT_NAME}ConfigVersion.cmake"
                                 VERSION ${PROJECT_VERSION}
                                 COMPATIBILITY SameMajorVersion)

install(TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}Targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

configure_package_config_file(
 "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
 "${PROJECT_BINARY_DIR}/cmake/${PROJECT_NAME}Config.cmake"
 INSTALL_DESTINATION
 ${PROJECT_BINARY_DIR}/cmake)

export(EXPORT ${PROJECT_NAME}Targets
  FILE "${PROJECT_BINARY_DIR}/cmake/${PROJECT_NAME}Targets.cmake"
  )

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include DESTINATION ${PROJECT_BINARY_DIR} FILES_MATCHING PATTERN *.hpp)

#
# Avoid incorrect warnings present in Clang and not in GCC
#
# See:
# https: // bugs.llvm.org/show_bug.cgi?id=21629
# https: // stackoverflow.com/questions/13905200/is-it-wise-to-ignore-gcc-clangs-wmissing-braces-warning
#
if(INSTALL_EXAMPLES OR INSTALL_TESTS)
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
     add_compile_options(-Wno-missing-braces)
  endif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
endif(INSTALL_EXAMPLES OR INSTALL_TESTS)

#
# Installation of the examples
#
if(INSTALL_EXAMPLES)

    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/examples)
    include_directories(include examples)
    file(GLOB EXAMPLE_SOURCES ${PROJECT_SOURCE_DIR}/examples/*.cpp)

    set(CMAKE_CXX_FLAGS "-O3 -pedantic -Wall -Wextra")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -pedantic -Wall -Wextra")

    foreach(examplesourcefile ${EXAMPLE_SOURCES})

      get_filename_component(examplename ${examplesourcefile} NAME_WE)
      add_executable(${examplename} ${examplesourcefile})
      set_target_properties(${examplename} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

      if(SAGA_CUDA_ENABLED)
          set_source_files_properties(${examplesourcefile} PROPERTIES LANGUAGE CUDA)
          set_property(TARGET ${examplename} PROPERTY CUDA_ARCHITECTURES OFF)
      endif(SAGA_CUDA_ENABLED)

    endforeach(examplesourcefile ${EXAMPLE_SOURCES})

endif(INSTALL_EXAMPLES)

#
# Installation of the test scripts
#
if(INSTALL_TESTS)

    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/test/cpp)
    include_directories(include test/include)

    file(GLOB TEST_SOURCES ${PROJECT_SOURCE_DIR}/test/cpp/*.cpp)

    set(CMAKE_CXX_FLAGS "-O3 -pedantic -Wall -Wextra")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -pedantic -Wall -Wextra")

    foreach(testsourcefile ${TEST_SOURCES})

      get_filename_component(testname ${testsourcefile} NAME_WE)
      add_executable(${testname} ${testsourcefile})
      set_target_properties(${testname} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/cpp CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

      if(SAGA_CUDA_ENABLED)
          set_source_files_properties(${testsourcefile} PROPERTIES LANGUAGE CUDA)
          set_property(TARGET ${testname} PROPERTY CUDA_ARCHITECTURES OFF)
      endif(SAGA_CUDA_ENABLED)

    endforeach(testsourcefile ${TEST_SOURCES})

    if(SAGA_CUDA_ENABLED)
       
      file(GLOB TEST_SOURCES ${PROJECT_SOURCE_DIR}/test/cuda/*.cpp)

      foreach(testsourcefile ${TEST_SOURCES})

        get_filename_component(testname ${testsourcefile} NAME_WE)
        add_executable(${testname} ${testsourcefile})
        set_target_properties(${testname} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/cuda CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

        set_source_files_properties(${testsourcefile} PROPERTIES LANGUAGE CUDA)
        set_property(TARGET ${testname} PROPERTY CUDA_ARCHITECTURES OFF)

      endforeach(testsourcefile ${TEST_SOURCES})
  
    endif(SAGA_CUDA_ENABLED)

endif(INSTALL_TESTS)
