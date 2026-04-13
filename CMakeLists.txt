cmake_minimum_required(VERSION 3.10)
project(MatrixMul_Lab2_MPI CXX)

set(CMAKE_CXX_STANDARD 11)  # Меняем с 17 на 11 для совместимости
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(MPI REQUIRED)

add_executable(lab3_matrix_mul_mpi
    lab_3.cpp
)

target_compile_definitions(lab3_matrix_mul_mpi PRIVATE
    PROJECT_DIR="${CMAKE_SOURCE_DIR}"
)

target_link_libraries(lab3_matrix_mul_mpi PRIVATE MPI::MPI_CXX)

# Убираем Windows-specific флаги
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR 
   CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(lab3_matrix_mul_mpi PRIVATE
        -Wall
        -Wextra
        -std=c++11
    )
endif()

set_target_properties(lab3_matrix_mul_mpi PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
)

message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "MPI found: ${MPI_CXX_FOUND}")