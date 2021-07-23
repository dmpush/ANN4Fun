#include <iostream>
#include <string>
#include <cassert>
#include <cmath>
#include <cstdlib>

#include <BackendOpenMP.hpp>
#include <SimpleTutor.hpp>
#include <Input.hpp>
#include <Model.hpp>
#include <Reshape.hpp>

using namespace std;

template<typename T>
void test1() {
    cout<<"Проверка Reshape<"<<typeid(T).name()<<">...";
    bool hasException{false};
    try {
	Model<T> model({2,3});
	model. template addLayer<Reshape<T>>({3,2});
	model.build(BackendOpenMP<T>::build());

	for(int i=0; i<1000; i++) {
	    T a=static_cast<T>(std::rand())/static_cast<T>(RAND_MAX);
	    T b=static_cast<T>(std::rand())/static_cast<T>(RAND_MAX);
	    T c=static_cast<T>(std::rand())/static_cast<T>(RAND_MAX);
	    T d=static_cast<T>(std::rand())/static_cast<T>(RAND_MAX);
	    T e=static_cast<T>(std::rand())/static_cast<T>(RAND_MAX);
	    T f=static_cast<T>(std::rand())/static_cast<T>(RAND_MAX);
	    model.batchBegin();

	    model.setInput(0, a);
	    model.setInput(1, b);
	    model.setInput(2, c);
	    model.setInput(3, d);
	    model.setInput(4, e);
	    model.setInput(5, f);
	    model.forward();

	    assert(std::abs(a-model.getOutput(0))<0.0001);
	    assert(std::abs(b-model.getOutput(1))<0.0001);
	    assert(std::abs(c-model.getOutput(2))<0.0001);
	    assert(std::abs(d-model.getOutput(3))<0.0001);
	    assert(std::abs(e-model.getOutput(4))<0.0001);
	    assert(std::abs(f-model.getOutput(5))<0.0001);

	    model.setOutput(0, a);
	    model.setOutput(1, b);
	    model.setOutput(2, c);
	    model.setOutput(3, d);
	    model.setOutput(4, e);
	    model.setOutput(5, f);
	    model.backward();
	    assert(std::abs(model.getInputErrors()->raw(0))<0.0001);
	    assert(std::abs(model.getInputErrors()->raw(1))<0.0001);
	    assert(std::abs(model.getInputErrors()->raw(2))<0.0001);
	    assert(std::abs(model.getInputErrors()->raw(3))<0.0001);
	    assert(std::abs(model.getInputErrors()->raw(4))<0.0001);
	    assert(std::abs(model.getInputErrors()->raw(5))<0.0001);

	    model.batchEnd();

	};
	assert(model.getInputs()->dims()[0]==2);
	assert(model.getInputs()->dims()[1]==3);
	assert(model.getOutputs()->dims()[0]==3);
	assert(model.getOutputs()->dims()[1]==2);
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
    cout<<"Ok."<<endl;
    return 0;
};

