# Файл — «CMakeLists.txt»
# Выбираем минимальную необходимую версию cmake
cmake_minimum_required(VERSION 3.13)
enable_testing()
# Присваиваем имя проекту
project(unit.Tensor2JPEG)
find_package(JPEG)
#add_definitions()
# Создаём цель (исполняемый файл)
include_directories(../../misc ../../ann4fun)
add_executable(
  ${PROJECT_NAME} # Имя файла
  main.cpp
)
set(HEADERS
)
target_link_libraries(${PROJECT_NAME} JPEG::JPEG)
add_test(
    NAME unit_arctan
    COMMAND ./${PROJECT_NAME}
)
