cmake_minimum_required(VERSION 3.31)
project(Project3)

set(CMAKE_CXX_STANDARD 20)

add_executable(Project3 src/main.cpp
        src/clustering/kmeans.cpp
        src/clustering/dbscan.cpp
        src/utils/csv_reader.cpp
        include/common.hpp)
