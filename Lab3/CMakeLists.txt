cmake_minimum_required(VERSION 3.15)
project(Lab3)

SET(CMAKE_CXX_COMPILER /usr//local/bin/mpicxx)

include_directories(/usr/local/opt/mpich/include)
link_directories(/usr/local/opt/mpich/lib)


add_executable(Lab3 main.cpp)
add_executable(Smart smartParall.cpp)
add_executable(Lab3NoParallel mainNoParallel.cpp)
add_executable(newLab newLab.cpp)
target_link_libraries(Lab3 mpi)
target_link_libraries(Smart mpi)
target_link_libraries(newLab mpi)
