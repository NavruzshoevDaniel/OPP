cmake_minimum_required(VERSION 3.15)
project(Lab4)

SET(CMAKE_CXX_COMPILER /usr//local/bin/mpicxx)

include_directories(/usr/local/opt/mpich/include)
link_directories(/usr/local/opt/mpich/lib)

add_executable(Lab4 main.cpp)
target_link_libraries(Lab4 mpi)