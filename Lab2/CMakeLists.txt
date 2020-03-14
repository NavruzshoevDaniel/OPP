cmake_minimum_required(VERSION 3.15)
project(Lab2)

SET(CMAKE_CXX_COMPILER /usr/local/bin/g++-9)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
include_directories(/usr/local/opt/libomp/include)
link_directories(/usr/local/opt/libomp/lib)

add_executable(Lab2 main.cpp )
add_executable(Lab2_1.1 mainOpenMP1.cpp)
add_executable(Lab2_1.2 mainOpenMP2.cpp)


target_link_libraries(Lab2 omp)
target_link_libraries(Lab2_1.1 omp)
target_link_libraries(Lab2_1.2 omp)
