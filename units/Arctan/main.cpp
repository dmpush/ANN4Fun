#include <iostream>
#include <string>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <vector>

#include <DataHolder.hpp>
#include <SimpleTutor.hpp>
#include <Input.hpp>
#include <Layer.hpp>
#include <Model.hpp>
#include <Arctan.hpp>
#include <ReLU.hpp>
#include <Gain.hpp>

template<typename T>
T OR(T x, T y) { return std::max(x,y); };
template<typename T>
T AND(T x, T y) { return std::min(x,y); };
template<typename T>
T NOT(T x) { return  - x; };
template<typename T>
T XOR(T x, T y) { return OR(AND(x, NOT(y)), AND(NOT(x), y)) ; };

using namespace std;




template<typename T>
bool taskXOR() {
    std::vector<float> x{-1.0, +1.0, -1.0, +1.0};
    std::vector<float> y{-1.0, -1.0, +1.0, +1.0};
    bool hasException{false};
    try {
	
	Model<T> model({2});
	model. template addLayer<Layer<T>>({7});
	model. template addLayer<Arctan<T>>();
	model. template addLayer<Layer<T>>({5});
	model. template addLayer<Arctan<T>>();
	model. template addLayer<Layer<T>>({3});
	model. template addLayer<Arctan<T>>();


	for(size_t i=0; i<1000; i++) {
	    model.batchBegin();
	    for(size_t k=0; k<4; k++) {
		model.setInput(0, x[k]);
		model.setInput(1, y[k]);
		model.forward();
		model.setOutput(0, AND(x[k], y[k])*0.9); //AND
		model.setOutput(1, OR (x[k], y[k])*0.9); //OR
		model.setOutput(2, XOR(x[k], y[k])*0.9); //XOR
		model.backward();
	    };
	    model.batchEnd();
	};
	// проверка
	for(size_t k=0; k<x.size(); k++) {
	    model.setInput(0, x[k]);
	    model.setInput(1, y[k]);
	    model.forward();
	    if(  std::abs(AND(x[k], y[k])-model.getOutput(0))>=0.3 )
		return false;
	    if(  std::abs(OR (x[k], y[k])-model.getOutput(1))>=0.3 )
		return false;
	    if(  std::abs(XOR(x[k], y[k])-model.getOutput(2))>=0.3 )
		return false;
	};
    } catch(std::runtime_error ex) {
	hasException=true;
	cout<<ex.what()<<endl;
    };
    if( hasException ) 
	return false;
//    cout<<"ok."<<endl;
    return true;
};

template<typename T>
void test2() {
    cout<<"Проверка Arctan<"<<typeid(T).name()<<"> на простых усилителях Gain...";
    bool hasException{false};
    try {
	Model<T> model({2});
	model. template addLayer<Gain<T>>();
	model. template addLayer<Arctan<T>>();
	for(int i=0; i<1000; i++) {
	    model.batchBegin();
	    model.setInput(0, +1.0);
	    model.setInput(1, -1.0);
	    model.forward();
	    model.setOutput(0, -0.314); 
	    model.setOutput(1, +0.271); 
	    model.backward();
	    model.batchEnd();
	};
	model.setInput(0, +1.0);
	model.setInput(1, -1.0);
	model.forward();
	assert(  std::abs(-0.314 - model.getOutput(0))<0.1 );
	assert(  std::abs(+0.271 - model.getOutput(1))<0.1 );

    
    } catch(std::runtime_error ex) {
	hasException=true;
	cout<<ex.what()<<endl;
    };
    assert(! hasException );
    cout<<"ok."<<endl;
};

template<typename T>
void test1() {
    cout<<"Проверка Arctan<"<<typeid(T).name()<<"> на логических функциях: ";
    size_t cnt=0;
    for(size_t k=0; k<100; k++)
	cnt += taskXOR<T>() ? 1u : 0u;
    std::cout<<"пройдено тестов "<<cnt<<" из 100...";
    assert(cnt>=50);
    std::cout<<"ok."<<std::endl;
};

template<typename T>
void test() {
    test1<T>();
    test2<T>();
};

int main()
{
    test<float>();
    test<double>();
    test<long double>();
    std::cout<<"Ok."<<std::endl;
    return 0;
};

