#include <iostream>
#include <string>
#include <cassert>
#include <cmath>
#include <cstdlib>

#include <DataHolder.hpp>
#include <SimpleTutor.hpp>
#include <Input.hpp>
#include <Model.hpp>
#include <Wire.hpp>

using namespace std;

template<typename T>
void test1() {
    cout<<"Проверка Wire<"<<typeid(T).name()<<">...";
    bool hasException{false};
    try {
	Model<T> model({2});
	model. template addLayer<Wire<T>>();


	for(int i=0; i<1000; i++) {
	    float x=static_cast<float>(std::rand())/static_cast<float>(RAND_MAX);
	    float y=static_cast<float>(std::rand())/static_cast<float>(RAND_MAX);
	    model.setMode(ANN<T>::TrainMode);

	    model.batchBegin();

	    model.setInput(0, x);
	    model.setInput(1, y);
	    model.forward();

	    assert(std::abs(x-model.getOutput(0))<0.0001);
	    assert(std::abs(y-model.getOutput(1))<0.0001);

	    model.setOutput(0, x);
	    model.setOutput(1, y);
	    model.backward();
	    assert(std::abs(model.getInputErrors()->raw(0))<0.0001);
	    assert(std::abs(model.getInputErrors()->raw(1))<0.0001);

	    model.batchEnd();


	};
    } catch(std::runtime_error ex) {
	hasException=true;
	cout<<ex.what()<<endl;
    };
    assert( ! hasException);
    cout<<"ok."<<endl;
};

template <typename T>
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

