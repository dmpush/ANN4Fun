#include <iostream>
#include <string>
#include <cassert>
#include <cmath>
#include <cstdlib>

#include <BackendOpenMP.hpp>
#include <SimpleTutor.hpp>
#include <Input.hpp>
#include <Model.hpp>
#include <Gain.hpp>

using namespace std;

template<typename T>
void test1() {
    cout<<"Проверка Gain<"<<typeid(T).name()<<">...";
    bool hasException{false};
    try {
	Model<T> model({2});
	model. template addLayer<Gain<T>>();
	model. template addLayer<Gain<T>>();
	model. template addLayer<Gain<T>>();
//	model. template addLayer<Gain<T>>();
	model.build(BackendOpenMP<T>::build());


	float x=3.14;//static_cast<float>(std::rand())/static_cast<float>(RAND_MAX);
	float y=2.71;//static_cast<float>(std::rand())/static_cast<float>(RAND_MAX);
	for(int i=0; i<1000; i++) {
	    model.batchBegin();

	    model.setInput(0, 1);
	    model.setInput(1, -1);
	    model.forward();
	
	    model.setOutput(0, x);
	    model.setOutput(1, y);
	    model.backward();
	    model.batchEnd();
	};
	model.setInput(0, 1);
	model.setInput(1, -1);
	model.forward();

	assert(std::abs(x-model.getOutput(0))<0.0001);
	assert(std::abs(y-model.getOutput(1))<0.0001);
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

