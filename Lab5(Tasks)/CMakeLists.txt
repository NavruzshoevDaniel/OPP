cmake_minimum_required(VERSION 3.15)
project(Lab5)

SET(CMAKE_CXX_COMPILER /usr//local/bin/mpicxx)

include_directories(/usr/local/opt/mpich/include)
link_directories(/usr/local/opt/mpich/lib)

add_executable(Lab5 main.cpp)
target_link_libraries(Lab5 mpi)