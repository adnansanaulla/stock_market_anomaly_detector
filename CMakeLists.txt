cmake_minimum_required(VERSION 3.31)
project(Project3)

set(CMAKE_CXX_STANDARD 20)

add_executable(Project3
        src/algs/anomoly_sliding_window.cpp
        src/algs/anomoly_heap.cpp
        src/utils/rolling_stats.cpp
        src/utils/csv_utils.cpp
        src/main.cpp)
