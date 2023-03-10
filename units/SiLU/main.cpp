#include <iostream>
#include <string>
#include <cassert>
#include <cmath>

#include <BackendOpenMP.hpp>
#include <SimpleTutor.hpp>
#include <Input.hpp>
#include <Layer.hpp>
#include <Model.hpp>
#include <SiLU.hpp>

using namespace std;

template<typename T>
bool task1() {
    bool hasException{false};
    try {
	Model<T> model({2});
	model. template addLayer<Layer<T>>({5});
	model. template addLayer<SiLU<T>>();
	model. template addLayer<Layer<T>>({3});
	model. template addLayer<SiLU<T>>();
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
	if(std::abs(1.0-model.getOutput(0))>=0.01)
	    return false;
	if(std::abs(2.0-model.getOutput(1))>=0.01)
	    return false;
	if(std::abs(3.0-model.getOutput(2))>=0.01)
	    return false;
	model.setInput(0, 1.0);
	model.setInput(1, 0.0);
	model.forward();
	if(std::abs(3.14-model.getOutput(0))>=0.01)
	    return false;
	if(std::abs(2.71-model.getOutput(1))>=0.01)
	    return false;
	if(std::abs(1.41-model.getOutput(2))>=0.01)
	    return false;
    } catch(std::runtime_error ex) {
	hasException=true;
	cout<<ex.what()<<endl;
    };
    if(  hasException)
	return false;
    return true;
};

template<typename T>
void test() {
    cout<<"Проверка SiLU<"<<typeid(T).name()<<"> на двухслойной сети: ";
    size_t cnt=0;
    for(size_t k=0; k<100; k++)
	cnt+=task1<T>() ? 1 : 0;
    std::cout<<cnt<<" успешных из 100..";
    assert(cnt>10);
    cout<<"ok."<<endl;
};

int main()
{
    test<float>();
    test<double>();
    test<long double>();
    cout<<"Ok."<<endl;
    return 0;
};

