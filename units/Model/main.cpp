#include <iostream>
#include <string>
#include <cassert>
#include <cmath>
#include <memory>

#include <BackendOpenMP.hpp>
#include <SimpleTutor.hpp>
#include <Input.hpp>
#include <Layer.hpp>
#include <Model.hpp>

using namespace std;

template<typename T>
void test1() {
    cout<<"Проверка Model<"<<typeid(T).name()<<"> на двухслойной сети без нелинейностей...";
    bool hasException{false};
    try {
	Model<T> model({2});
	model. template addLayer<Layer<T>>({2});
	model. template addLayer<Layer<T>>({3});
	model.build(BackendOpenMP<T>::build());

	model(1)->setTutor(std::make_unique<SimpleTutor<T>>(0.1));

	for(int i=0; i<1000; i++) {
	    model.batchBegin();

	    model.setInput(0, 0.0);
	    model.setInput(1, 1.0);
	    model.forward();
	    model.setOutput(0, 1.0);
	    model.setOutput(1, 2.0);
	    model.setOutput(2, 3.0);
	    model.backward();

	    model.setInput(0, 1.0);
	    model.setInput(1, 0.0);
	    model.forward();
	    model.setOutput(0, 3.14);
	    model.setOutput(1, 2.71);
	    model.setOutput(2, 1.41);
	    model.backward();

	    model.batchEnd();

	};
	model.setInput(0, 0.0);
	model.setInput(1, 1.0);
	model.forward();
	assert(std::abs(1.0-model.getOutput(0))<0.01);
	assert(std::abs(2.0-model.getOutput(1))<0.01);
	assert(std::abs(3.0-model.getOutput(2))<0.01);
	model.setInput(0, 1.0);
	model.setInput(1, 0.0);
	model.forward();
	assert(std::abs(3.14-model.getOutput(0))<0.01);
	assert(std::abs(2.71-model.getOutput(1))<0.01);
	assert(std::abs(1.41-model.getOutput(2))<0.01);
    } catch(std::runtime_error ex) {
	hasException=true;
	cout<<ex.what()<<endl;
    };
    assert( ! hasException);
    cout<<"ok."<<endl;
};


template<typename T>
void testComposition() {
    cout<<"Проверка композиции Model<"<<typeid(T).name()<<">...";
    bool hasException{false};
    try {
	std::vector<size_t> inputShape={2};
	auto model1 = std::make_shared<Model<T>>(inputShape);
	model1-> template addLayer<Layer<T>>({2});
	Model<T> model(model1);
	model. template addLayer<Layer<T>>({3});
	model.build(BackendOpenMP<T>::build());

	for(int i=0; i<1000; i++) {
	    model.batchBegin();

	    model.setInput(0, 0.0);
	    model.setInput(1, 1.0);
//            model.setInput(2, 0.5);
	    model.forward();
	    model.setOutput(0, 1.0);
	    model.setOutput(1, 2.0);
	    model.setOutput(2, 3.0);
	    model.backward();

	    model.setInput(0, 1.0);
	    model.setInput(1, 0.0);
	    model.forward();
	    model.setOutput(0, 3.14);
	    model.setOutput(1, 2.71);
	    model.setOutput(2, 1.41);
	    model.backward();

	    model.batchEnd();

	};
	model.setInput(0, 0.0);
	model.setInput(1, 1.0);
	model.forward();
	assert(std::abs(1.0-model.getOutput(0))<0.01);
	assert(std::abs(2.0-model.getOutput(1))<0.01);
	assert(std::abs(3.0-model.getOutput(2))<0.01);
	model.setInput(0, 1.0);
	model.setInput(1, 0.0);
	model.forward();
	assert(std::abs(3.14-model.getOutput(0))<0.01);
	assert(std::abs(2.71-model.getOutput(1))<0.01);
	assert(std::abs(1.41-model.getOutput(2))<0.01);
    } catch(std::runtime_error ex) {
	hasException=true;
	cout<<ex.what()<<endl;
    };
    assert( ! hasException);
    cout<<"ok."<<endl;
};

template<typename T>
void test() {
    test1<T>();
    testComposition<T>();
};

int main()
{
    test<float>();
//    test<double>();
//    test<long double>();
    cout<<"Ok."<<endl;
    return 0;
};

