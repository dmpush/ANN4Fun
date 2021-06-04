#include <iostream>
#include <string>
#include <cassert>
#include <stdexcept>
#include <memory>
#include <vector>

#include <Input.hpp>
#include <Layer.hpp>
#include <DataHolder.hpp>

using namespace std;

template<typename T>
void exceptMsg(std::runtime_error e) {
	cout<<"    Перехвачено<"<<typeid(T).name()<<">: \""<<e.what()<<"\" ok."<<endl;
};


template<typename T>
void testExceptions() {
    bool hasException{false};


    cout<<"Проверка Input на исключения..."<<endl;
    hasException=false;
    try {
	auto  inputs=std::make_shared<Input<T>> (std::vector<size_t>({2,2}));
	auto  layer=std::make_shared<Layer<T>> (inputs.get(), std::vector<size_t>({2}));
    } catch(std::runtime_error e) {
	exceptMsg<T>(e);
	hasException=true;
    };
    assert(hasException);

    hasException=false;
    try {
	auto  inputs=std::make_shared<Input<T>> (std::vector<size_t>({2}));
	auto  layer=std::make_shared<Layer<T>> (inputs.get(), std::vector<size_t>({2,2}));
    } catch(std::runtime_error e) {
	exceptMsg<T>(e);
	hasException=true;
    };
    assert(hasException);

};



template<typename T>
void test() {
    testExceptions<T>();
};



int main()
{
    test<float>();
    test<double>();
    test<long double>();
    return 0;
};

