#include <iostream>
#include <string>
#include <cassert>
#include <cmath>

#include <DataHolder.hpp>
#include <SimpleTutor.hpp>
#include <Input.hpp>
#include <Layer.hpp>
#include <Model.hpp>
#include <SoftMax.hpp>

using namespace std;

template<typename T>
void test1() {
    cout<<"Проверка SoftMax<"<<typeid(T).name()<<"> на однослойной сети...";
    bool hasException{false};
    T ex1[]={0.9, 0.05,0.05};
    T ex2[]={0.0, 0.9, 0.1};
    try {
	Model<T> model({2});
	model. template addLayer<Layer<T>>({3});
	model. template addLayer<SoftMax<T>>();

	model(1)->setTutor(std::make_unique<SimpleTutor<T>>(0.1));

	for(int i=0; i<10000; i++) {
	    model.setMode(ANN<T>::TrainMode);

	    model.batchBegin();

	    model.setInput(0, 0.0);
	    model.setInput(1, 1.0);
	    model.forward();
	    model.setOutput(0, ex1[0]);
	    model.setOutput(1, ex1[1]);
	    model.setOutput(2, ex1[2]);
	    model.backward();

	    model.setInput(0, 1.0);
	    model.setInput(1, 0.0);
	    model.forward();
	    model.setOutput(0, ex2[0]);
	    model.setOutput(1, ex2[1]);
	    model.setOutput(2, ex2[2]);
	    model.backward();

	    model.batchEnd();

	};
	model.setMode(ANN<T>::WorkMode);
	model.setInput(0, 0.0);
	model.setInput(1, 1.0);
	model.forward();
	assert(std::abs(ex1[0]-model.getOutput(0))<0.1);
	assert(std::abs(ex1[1]-model.getOutput(1))<0.1);
	assert(std::abs(ex1[2]-model.getOutput(2))<0.1);
	model.setInput(0, 1.0);
	model.setInput(1, 0.0);
	model.forward();
	assert(std::abs(ex2[0]-model.getOutput(0))<0.1);
	assert(std::abs(ex2[1]-model.getOutput(1))<0.1);
	assert(std::abs(ex2[2]-model.getOutput(2))<0.1);
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
};

int main()
{
    test<float>();
    test<double>();
    test<long double>();
    return 0;
};

