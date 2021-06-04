#include <iostream>
#include <string>
#include <cassert>
#include <cmath>

#include <DataHolder.hpp>
#include <SimpleTutor.hpp>
#include <Input.hpp>
#include <Layer.hpp>
#include <Model.hpp>

using namespace std;

template<typename T>
void test1() {
    cout<<"Проверка Model на двухслойной сети без нелинейностей...";
    bool hasException{false};
    try {
	Model<T> model({2});
	model. template addLayer<Layer<T>>({2});
	model. template addLayer<Layer<T>>({3});

	model(1)->setTutor(std::make_unique<SimpleTutor<T>>(0.1));

	for(int i=0; i<1000; i++) {
	    model.setMode(ANN<T>::TrainMode);

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
	model.setMode(ANN<T>::WorkMode);
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

int main()
{
    test1<float>();
    return 0;
};

