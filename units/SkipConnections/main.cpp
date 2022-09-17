#include <iostream>
#include <string>
#include <cassert>
#include <stdexcept>
#include <memory>
#include <vector>
#include <cmath>

#include <Input.hpp>
#include <Layer.hpp>
#include <SkipConnections.hpp>
#include <Assertion.hpp>
#include <ANN.hpp>
#include <TestRot3.hpp>
#include <BackendOpenMP.hpp>
#include <SimpleTutor.hpp>
using namespace std;

template<typename T>
void testTrain() {
    cout<<"Проверка SkipConnections<"<<typeid(T).name()<<"> обучаемости на СЛАУ ...";

	std::vector<size_t> inputShape={2};
	std::vector<size_t> outputShape{2};
	auto con=std::make_shared<SkipConnections<float>>( inputShape );
	con->getModel()->template addLayer<Layer<float>> ( outputShape );
	con->build(BackendOpenMP<float>::build());
	for(size_t n=0; n<10000; n++) {
	    con->batchBegin();
	    con->setInput(0, 1.0);
	    con->setInput(1, 0.0);
	    con->forward();
//	    std::cout<<con->getOutput(0)<<", "<<con->getOutput(1)<<std::endl;
	    con->setOutput(0, M_PI);
	    con->setOutput(1, M_E);
	    con->backward();

	    con->setInput(0, 0.0);
	    con->setInput(1, 1.0);
	    con->forward();
//	    std::cout<<con->getOutput(0)<<", "<<con->getOutput(1)<<std::endl;
	    con->setOutput(0, M_E);
	    con->setOutput(1, M_PI);
	    con->backward();

	    con->batchEnd();
	};
	con->setInput(0, 1.0);
	con->setInput(1, 0.0);
	con->forward();
	assert(std::abs(M_PI-con->getOutput(0))<1e-3);
	assert(std::abs(M_E-con->getOutput(1))<1e-3);

	con->setInput(0, 0.0);
	con->setInput(1, 1.0);
	con->forward();
	assert(std::abs(M_E-con->getOutput(0))<1e-3);
	assert(std::abs(M_PI-con->getOutput(1))<1e-3);
	cout<<"ok."<<endl;

};


int main() {

    testTrain<float>();
    testTrain<double>();
    testTrain<long double>();
    return 0;
};


