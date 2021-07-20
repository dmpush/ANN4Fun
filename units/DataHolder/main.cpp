#include <iostream>
#include <string>
#include <cassert>
#include <stdexcept>
#include <memory>
#include <cmath>
#include <DataHolder.hpp>
#include <Timer.hpp>
using namespace std;

template<typename T>
void exceptMsg(std::runtime_error e) {
	cout<<"    Перехвачено<"<<typeid(T).name()<<">: \""<<e.what()<<"\" ok."<<endl;
};


template<typename T>
void testExceptions() {
    bool hasException{false};

    auto holder1=std::make_shared<DataHolder<T>>();
    holder1->append("T", {2,2});
    holder1->append("U", {1,2,3,4,5});
    holder1->build();

    cout<<"Проверка геттера переменных на исключения..."<<endl;
    hasException=false;
    try {
	holder1->get("F");
    } catch(std::runtime_error e) {
	exceptMsg<T>(e);
	hasException=true;
    };
    assert(hasException);

    cout<<"Проверка геттера на исключения..."<<endl;
    hasException=false;
    try {
	holder1->get("T")->val({1,1,1});
    } catch(std::runtime_error e) {
	exceptMsg<T>(e);
	hasException=true;
    };
    assert(hasException);


    cout<<"Проверка cеттера на исключения..."<<endl;
    hasException=false;
    try {
	holder1->get("T")->val({1,1,1}) = 1;
    } catch(std::runtime_error e) {
	exceptMsg<T>(e);
	hasException=true;
    };
    assert(hasException);

    auto holder2=std::make_shared<DataHolder<T>>();
    holder2->append("X", {4,4});
    holder2->build();


};

template<typename T>
void testGet() {
    cout<<"Проверка добавления/извлечения тензоров из хранилища...";
    DataHolder<T> holder1;
    holder1.append("A", {2,3});
    holder1.append("B", {2,3,4});
    holder1.append("C", {15});
    holder1.build();

    assert(holder1.get("A")->dim()==2);
    assert(holder1.get("B")->dim()==3);
    assert(holder1.get("C")->dim()==1);

    auto dimA=holder1.get("A")->dims();
    auto dimB=holder1.get("B")->dims();
    auto dimC=holder1.get("C")->dims();
    assert(dimA[0]==2);
    assert(dimA[1]==3);

    assert(dimB[0]==2);
    assert(dimB[1]==3);
    assert(dimB[2]==4);

    assert(dimC[0]==15);
    cout<<"ok."<<endl;
};


template<typename T>
void testClone() {
    cout<<"Проверка операции клонирования...";
    auto holder1=std::make_shared<DataHolder<T>>();
    holder1->append("A", {2,3});
    holder1->append("B", {2,3,4});
    holder1->append("C", {15});
    holder1->build();
    for(size_t i=0; i<holder1->size(); i++)
	holder1->raw(i)=static_cast<T>(i);

    auto holder2 = holder1->clone();

    assert(holder1->size() == holder2->size());


    for(size_t i=0; i<2; i++)
	for(size_t j=0; j<3; j++)
	    assert(holder1->get("A")->val({i,j}) == holder2->get("A")->val({i,j}) );

    for(size_t i=0; i<2; i++)
	for(size_t j=0; j<3; j++)
	    for(size_t k=0; k<4; k++)
	    assert(holder1->get("B")->val({i,j,k}) == holder2->get("B")->val({i,j,k}) );

    for(size_t i=0; i<15; i++)
	    assert(holder1->get("C")->val({i}) == holder2->get("C")->val({i}) );


    cout<<"ok."<<endl;
};


template<typename T>
void testFill() {
    cout<<"Проверка операции fill()...";
    auto holder1=std::make_shared<DataHolder<T>>();
    holder1->append("A", {3,4});
    holder1->build();
    holder1->fill(static_cast<T>(3.1415f));
    for(size_t i=0; i<3; i++)
	for(size_t j=0; j<4; j++)
	    assert( std::abs(static_cast<double>(holder1->get("A")->val({i,j})) - 3.1415)<0.01 );
    cout<<"ok."<<endl;
};

template<typename T, size_t N>
void testMul() {
    cout<<"Проверка операции mul()...";
    auto holder=std::make_shared<DataHolder<T>>();
    holder->append("A", {N});
    holder->append("E", {N,N});
    holder->append("A1", {N});
    holder->build();
    auto A=holder->get("A");
    auto A1=holder->get("A1");
    auto E=holder->get("E");
    for(size_t i=0; i<N; i++)
	for(size_t j=0; j<N; j++)
	    E->val({i,j}) = i==j ? 1.0 : 0.0;
    A->uniformNoise(-1.0, +1.0);
    Timer timer;
    timer.tic();
    A1->mul(A,E);
    cout<<endl;
    cout<<"\t вектор на единичную матрицу "<<N<<"x"<<N<<" ...";
    cout<<" ("<<timer.toc()<<" sec) ";
    for(size_t i=0; i<N; i++)
	assert(std::abs(A->raw(i) - A1->raw(i))<1e-3);
    cout<<"ok"<<endl;
    cout<<"\t единичная матрица на вектор "<<N<<"x"<<N<<" ...";
    timer.tic();
    A1->mul(E,A);
    cout<<" ("<<timer.toc()<<" sec) ";
    for(size_t i=0; i<N; i++)
	assert(std::abs(A->raw(i) - A1->raw(i))<1e-3);
    cout<<"ok"<<endl;
    
}




template<typename T>
void test() {
    testGet<T>();
    testClone<T>();
    testExceptions<T>();
    testFill<T>();
    testMul<T, 10>();
    testMul<T, 100>();
    testMul<T, 1000>();
};



int main()
{
    test<float>();
    test<double>();
    test<long double>();
    return 0;
};

