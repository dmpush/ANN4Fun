cmake_minimum_required(VERSION 3.13)
project(units)
set(HEADERS
MNIST.hpp
)

add_subdirectory(./MNIST.classifier)
add_subdirectory(./MNIST.autoencoder)
add_subdirectory(./MNIST.GAN)
add_subdirectory(./fuzzy2jpeg)
