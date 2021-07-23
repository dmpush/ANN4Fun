#include <iostream>
#include <string>
#include <cassert>
#include <stdexcept>
#include <memory>
#include <vector>
#include <cstdlib>
#include <cmath>

#include <Input.hpp>
#include <BackendOpenMP.hpp>

using namespace std;

template<typename T>
void exceptMsg(std::runtime_error e) {
	cout<<"    Перехвачено<"<<typeid(T).name()<<">: \""<<e.what()<<"\" ok."<<endl;
};


template<typename T>
void test1() {
    bool hasException{false};

    cout<<"Проверка Input<"<<typeid(T).name()<<">...";


    hasException=false;
    try {
	auto  inputs=std::make_shared<Input<T>> ( std::vector<size_t>({2}) );
	inputs->build(BackendOpenMP<T>::build());
	T x=static_cast<T>(std::rand())/static_cast<T>(RAND_MAX);
	T y=static_cast<T>(std::rand())/static_cast<T>(RAND_MAX);
	inputs->batchBegin();
	inputs->setInput(0, x);
	inputs->setInput(1, y);
	inputs->forward();
	assert( std::abs(x - inputs->getOutput(0)) < std::max(x,y)*1e-4 );
	assert( std::abs(y - inputs->getOutput(1)) < std::max(x,y)*1e-4 );
	inputs->setOutput(0, T(0));
	inputs->setOutput(1, T(0));
	inputs->backward();
	assert( std::abs(-x - inputs->getInputErrors()->raw(0)) < std::max(x,y)*1e-4 );
	assert( std::abs(-y - inputs->getInputErrors()->raw(1)) < std::max(x,y)*1e-4 );
	inputs->batchEnd();
	
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

