cmake_minimum_required(VERSION 3.15)
project(Lab1_3)


SET(CMAKE_C_COMPILER /usr//local/bin/mpicc)
SET(CMAKE_CXX_COMPILER /usr//local/bin/mpic++)

include_directories(/usr/local/opt/mpich/include)
link_directories(/usr/local/opt/mpich/lib)

add_executable(Lab1.3_var_1 main_var1.cpp)
add_executable(Lab1 main.cpp)
add_executable(Lab1.3_var_2 main_var2.cpp)

target_link_libraries (Lab1.3_var_1 mpi)
target_link_libraries (Lab1.3_var_2 mpi)
target_link_libraries (Lab1 mpi)