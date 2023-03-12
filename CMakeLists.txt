# Файл — «CMakeLists.txt»
# Выбираем минимальную необходимую версию cmake
cmake_minimum_required(VERSION 3.13)
# Присваиваем имя проекту
set (CMAKE_CXX_STANDARD 20)
#set (CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -Werror")
enable_testing()
add_library(ann4fun INTERFACE)
target_link_libraries(ann4fun INTERFACE)
add_library(misc INTERFACE)
target_link_libraries(misc INTERFACE)

project(ANN4Fun)
find_package(OpenMP REQUIRED)
add_subdirectory(units)
add_subdirectory(examples)

