#include <iostream>
#include <string>
#include <cassert>
#include <stdexcept>
#include <memory>
#include <vector>
#include <cstdlib>
#include <cmath>

#include <Generator.hpp>
#include <DataHolder.hpp>

using namespace std;

template<typename T>
void exceptMsg(std::runtime_error e) {
	cout<<"    Перехвачено<"<<typeid(T).name()<<">: \""<<e.what()<<"\" ok."<<endl;
};


template<typename T>
void test1() {
    bool hasException{false};

    cout<<"Проверка Generator<"<<typeid(T).name()<<">...";


    hasException=false;
    try {
	auto  gen=std::make_shared<Generator<T>> ( std::vector<size_t>({10}) );
	assert(gen->getInputs()->dim() == 0);
	assert(gen->getNumInputs() == 0);
	assert(gen->getNumOutputs() == 10);
	gen->batchBegin();
	T m1{0.0f}, m2{0.0f};
	for(size_t smp=0; smp<1000; smp++) {
	    gen->forward();
	    for(size_t o=0; o<10; o++) {
		auto y=gen->getOutput(o);
		m1 += y;
		m2 += y*y;
		gen->backward();
	    };
	};
	gen->batchEnd();
	m1/=static_cast<T>(10000);
	m2/=static_cast<T>(10000);
	assert(std::abs(static_cast<float>(m1) - 0.0f)<0.1f);
	assert(std::abs(static_cast<float>(m2) - 1.0f)<0.1f);
    } catch(std::runtime_error e) {
	exceptMsg<T>(e);
	hasException=true;
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
    cout<<"Ok."<<endl;
    return 0;
};

