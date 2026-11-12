cmake_minimum_required(VERSION 3.24)

# Honor an explicit -D override; the local default is /usr/local/cuda.
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "CUDA compiler")
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "GPU architectures")
endif()
project(leetgpu_practice LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Keep CUDA -I flags visible to editor compilation-database parsers.
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
find_package(CUDAToolkit REQUIRED)
include(CTest)

add_library(practice_common INTERFACE)
target_include_directories(practice_common INTERFACE "${CMAKE_SOURCE_DIR}/common")
target_link_libraries(practice_common INTERFACE CUDA::cudart)
target_compile_options(practice_common INTERFACE
    "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>:-lineinfo>"
    "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>"
)

function(add_exercise name)
    add_executable(${name} main.cpp solution.cu)
    target_link_libraries(${name} PRIVATE practice_common)
    if(BUILD_TESTING)
        add_test(NAME scaffold_${name} COMMAND ${name} --self-test)
        set_tests_properties(scaffold_${name} PROPERTIES LABELS scaffold RESOURCE_LOCK gpu TIMEOUT 180)
        add_test(NAME solution_${name} COMMAND ${name} --check-only)
        set_tests_properties(solution_${name} PROPERTIES LABELS solution RESOURCE_LOCK gpu TIMEOUT 180)
    endif()
endfunction()

add_subdirectory(stage1_cuda_basics)
add_subdirectory(stage2_llm_operators)
add_subdirectory(stage3_attention)

if(BUILD_TESTING)
    add_executable(reference_tests tests/reference_tests.cpp)
    target_include_directories(reference_tests PRIVATE common)
    add_test(NAME reference_math COMMAND reference_tests)
    set_tests_properties(reference_math PROPERTIES LABELS scaffold)
    add_executable(harness_tests tests/harness_tests.cu)
    target_link_libraries(harness_tests PRIVATE practice_common)
    add_test(NAME harness_contract COMMAND harness_tests)
    set_tests_properties(harness_contract PROPERTIES LABELS scaffold RESOURCE_LOCK gpu TIMEOUT 60)
endif()
