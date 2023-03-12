#include <iostream>
#include <string>
#include <cassert>
#include <stdexcept>
#include <memory>
#include <cmath>

#include <BackendOpenMP.hpp>
#include <MNIST.hpp>
#include <Tensor2JPEG.hpp>

using namespace std;



int main()
{
    MNIST<> mnist("../../../../");
    auto factory=BackendOpenMP<float>::build();
    auto dh=factory->makeHolderS();
    dh->append("X", {28,28});
    dh->build();
    auto X=dh->get("X");
    for(size_t s=0; s<10; s++) {
	auto t=mnist.getTrainSet()->getRandomSample();
	for(size_t i=0; i<X->size(); i++)
	    X->raw(i)= t->raw(i);
	tensormath::toJPEG<float>(X, "test-"+std::to_string(s)+".jpg");
    };

    return 0;
};

