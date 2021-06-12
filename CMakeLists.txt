cmake_minimum_required(VERSION 3.19)
project(biip_parallel)

set(CMAKE_CXX_STANDARD 14)

add_executable(biip_parallel biip.cpp)

include_directories($ENV{GUROBI_HOME}/include/)
find_library(GUROBI_LIB NAMES gurobi91 gurobi_c++ PATHS $ENV{GUROBI_HOME}/bin/)
target_link_libraries(biip_parallel ${GUROBI_LIB} pthread m)


