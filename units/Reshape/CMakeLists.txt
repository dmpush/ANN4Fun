# Файл — «CMakeLists.txt»
# Выбираем минимальную необходимую версию cmake
cmake_minimum_required(VERSION 3.13)
enable_testing()
# Присваиваем имя проекту
project(unit.Reshape)
#add_definitions()
# Создаём цель (исполняемый файл)
include_directories(../../misc ../../ann4fun)
add_executable(
  ${PROJECT_NAME} # Имя файла
  main.cpp
)
set(HEADERS
)
#target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
add_test(
    NAME unit_reshape
    COMMAND ./${PROJECT_NAME}
)
