#include <iostream>
#include <string>
#include <cassert>
#include <stdexcept>
#include <memory>
#include <vector>
#include <cmath>

#include <Input.hpp>
#include <Layer.hpp>
#include <ANN.hpp>

using namespace std;

template<typename T>
void exceptMsg(std::runtime_error e) {
	cout<<"    Перехвачено<"<<typeid(T).name()<<">: \""<<e.what()<<"\" ok."<<endl;
};


template<typename T>
void testExceptions() {
    bool hasException{false};


    cout<<"Проверка Layer<"<<typeid(T).name()<<"> на исключения..."<<endl;
    hasException=false;
    try {
	auto  inputs=std::make_shared<Input<T>> (std::vector<size_t>({2,2}));
	auto  layer=std::make_shared<Layer<T>> (inputs.get(), std::vector<size_t>({2}));
    } catch(std::runtime_error e) {
	exceptMsg<T>(e);
	hasException=true;
    };
    assert(hasException);

    hasException=false;
    try {
	auto  inputs=std::make_shared<Input<T>> (std::vector<size_t>({2}));
	auto  layer=std::make_shared<Layer<T>> (inputs.get(), std::vector<size_t>({2,2}));
    } catch(std::runtime_error e) {
	exceptMsg<T>(e);
	hasException=true;
    };
    assert(hasException);


};



template<typename T>
void testTrain() {
    cout<<"Проверка Layer<"<<typeid(T).name()<<"> обучаемости (детерминированная)...";
    bool hasException{false};
    try {
	auto  input=std::make_shared<Input<T>> ( std::vector<size_t>({2}) );
	auto  layer=std::make_shared<Layer<T>> ( input.get(), std::vector<size_t>({2}) );

	for(size_t it=0; it<1000; it++) {
	    input->batchBegin();
	    layer->batchBegin();
	    input->setInput(0, T(1));
	    input->setInput(1, T(0));
	    input->forward();
	    layer->forward();
	    layer->setOutput(0, T(3.14));
	    layer->setOutput(1, T(2.71));
	    layer->backward();
	    input->backward();

	    input->setInput(0, T(0));
	    input->setInput(1, T(1));
	    input->forward();
	    layer->forward();
	    layer->setOutput(0, T(2.71));
	    layer->setOutput(1, T(3.14));
	    layer->backward();
	    input->backward();

	    layer->batchEnd();
	    input->batchEnd();
	};
	input->setInput(0, T(1));
	input->setInput(1, T(0));
	input->forward();
	layer->forward();
	assert(std::abs(3.14 - layer->getOutput(0))<0.01);
	assert(std::abs(2.71 - layer->getOutput(1))<0.01);

	input->setInput(0, T(0));
	input->setInput(1, T(1));
	input->forward();
	layer->forward();
	assert(std::abs(2.71 - layer->getOutput(0))<0.01);
	assert(std::abs(3.14 - layer->getOutput(1))<0.01);
    } catch(std::runtime_error e) {
	exceptMsg<T>(e);
	hasException=true;
    };
    assert( ! hasException);
    cout<<"ok."<<endl;
};


template<typename T>
void test() {
    testExceptions<T>();
    testTrain<T>();
};



int main()
{
    test<float>();
    test<double>();
    test<long double>();
    return 0;
};

