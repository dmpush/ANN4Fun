cmake_minimum_required(VERSION 3.13)
project(units)
set(HEADERS
Console.hpp
ModelTest.hpp
OnlineStatistics.hpp
TestGains.hpp
TestRot3.hpp
TestXOR.hpp
Timer.hpp
)
add_subdirectory(DataHolder)
add_subdirectory(Input)
add_subdirectory(Layer)
add_subdirectory(Model)
add_subdirectory(Tutors)
add_subdirectory(Wire)
add_subdirectory(Arctan)
add_subdirectory(ReLU)
add_subdirectory(ReLUx)
add_subdirectory(ELU)
add_subdirectory(SELU)
add_subdirectory(SiLU)
add_subdirectory(SoftMax)
add_subdirectory(Gain)
add_subdirectory(Dropout)
add_subdirectory(Reshape)
add_subdirectory(Generator)
add_subdirectory(SkipConnections)
add_subdirectory(Tensor2JPEG)
add_subdirectory(MNIST)


