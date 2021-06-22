cmake_minimum_required(VERSION 3.19)
project(biip_parallel)

set(CMAKE_CXX_STANDARD 14)

add_executable(biip_parallel biip.cpp)

include_directories($ENV{GUROBI_HOME}/include/)
find_library(GUROBI_LIB NAMES gurobi91 gurobi_c++ PATHS $ENV{GUROBI_HOME}/bin/)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(biip_parallel PUBLIC OpenMP::OpenMP_CXX ${GUROBI_LIB} pthread m)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp")
set(OMP_THREAD_LIMIT 8)
