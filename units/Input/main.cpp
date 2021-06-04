#include <iostream>
#include <string>
#include <cassert>
#include <stdexcept>
#include <memory>
#include <vector>

#include <Input.hpp>
#include <DataHolder.hpp>

using namespace std;

template<typename T>
void exceptMsg(std::runtime_error e) {
	cout<<"    Перехвачено<"<<typeid(T).name()<<">: \""<<e.what()<<"\" ok."<<endl;
};


template<typename T>
void testExceptions() {
    bool hasException{false};

    auto  inputs=std::make_shared<Input<T>> (std::vector<size_t>({2}));

    cout<<"Проверка Input на исключения..."<<endl;
    hasException=false;
    try {
	inputs->batchBegin();
	inputs->setInput(0,T(0.0));
	inputs->setInput(1,T(1.0));
	inputs->forward();
	inputs->setOutput(0, T(3.14));
	inputs->setOutput(1, T(2.71));
	inputs->backward();
	inputs->batchEnd();
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

