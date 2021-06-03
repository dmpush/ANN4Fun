#include <iostream>
#include <string>
#include <cassert>
#include <stdexcept>
#include <DataHolder.hpp>

using namespace std;

template<typename T>
void exceptMsg(std::runtime_error e) {
	cout<<"    Перехвачено<"<<typeid(T).name()<<">: \""<<e.what()<<"\" ok."<<endl;
};


template<typename T>
void testExceptions() {
    bool hasException{false};

    DataHolder<T> holder1;
    holder1.append("T", {2,2});
    holder1.append("U", {1,2,3,4,5});
    holder1.build();

    cout<<"Проверка геттера на исключения..."<<endl;
    hasException=false;
    try {
	holder1.get("T")->get({1,1,1});
    } catch(std::runtime_error e) {
	exceptMsg<T>(e);
	hasException=true;
    };
    assert(hasException);

    hasException=false;
    try {
	holder1.get("U")->get({1,1,1,1,1});
    } catch(std::runtime_error e) {
	exceptMsg<T>(e);
	hasException=true;
    };
    assert(hasException);

    cout<<"Проверка cеттера на исключения..."<<endl;
    hasException=false;
    try {
	holder1.get("T")->set({1,1,1}, 1);
    } catch(std::runtime_error e) {
	exceptMsg<T>(e);
	hasException=true;
    };
    assert(hasException);

    hasException=false;
    try {
	holder1.get("U")->set({1,1,1,1,1}, 1);
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

