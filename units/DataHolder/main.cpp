#include <iostream>
#include <string>
#include <cassert>
#include <stdexcept>
#include <memory>
#include <cmath>
#include <IDataHolder.hpp>
#include <IBackendFactory.hpp>
#include <BackendOpenMP.hpp>

#include <Timer.hpp>
#include <Console.hpp>
using namespace std;

template<typename T>
void exceptMsg(std::runtime_error e) {
	cout<<"    Перехвачено<"<<typeid(T).name()<<">: \""<<e.what()<<"\" ok."<<endl;
};


template<typename T>
void testExceptions(typename IBackendFactory<T>::sPtr factory) {
    bool hasException{false};

    auto holder1=factory->makeHolderS();
    holder1->append("T", {2,2});
    holder1->append("U", {1,2,3,4,5});
    holder1->build();

    cout<<"Проверка геттера переменных на исключения..."<<endl;
    hasException=false;
    try {
	holder1->get("F");
    } catch(std::runtime_error& e) {
	exceptMsg<T>(e);
	hasException=true;
    };
    assert(hasException);

    cout<<"Проверка геттера на исключения..."<<endl;
    hasException=false;
    try {
	holder1->get("T")->val({1,1,1});
    } catch(std::runtime_error& e) {
	exceptMsg<T>(e);
	hasException=true;
    };
    assert(hasException);


    cout<<"Проверка cеттера на исключения..."<<endl;
    hasException=false;
    try {
	holder1->get("T")->val({1,1,1}) = 1;
    } catch(std::runtime_error& e) {
	exceptMsg<T>(e);
	hasException=true;
    };
    assert(hasException);

    auto holder2=factory->makeHolderS();
    holder2->append("X", {4,4});
    holder2->build();


};

template<typename T>
void testGet(typename IBackendFactory<T>::sPtr factory) {
    cout<<"Проверка добавления/извлечения тензоров из хранилища...";
    auto  holder1=factory->makeHolderS();
    holder1->append("A", {2,3});
    holder1->append("B", {2,3,4});
    holder1->append("C", {15});
    holder1->build();

    assert(holder1->get("A")->dim()==2);
    assert(holder1->get("B")->dim()==3);
    assert(holder1->get("C")->dim()==1);

    auto dimA=holder1->get("A")->dims();
    auto dimB=holder1->get("B")->dims();
    auto dimC=holder1->get("C")->dims();
    assert(dimA[0]==2);
    assert(dimA[1]==3);

    assert(dimB[0]==2);
    assert(dimB[1]==3);
    assert(dimB[2]==4);

    assert(dimC[0]==15);
    cout<<"ok."<<endl;
};


template<typename T>
void testClone(typename IBackendFactory<T>::sPtr factory) {
    cout<<"Проверка операции клонирования...";
    auto holder1=factory->makeHolderS();
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
void testFill(typename IBackendFactory<T>::sPtr factory) {
    cout<<"Проверка операции fill()...";
    auto holder1=factory->makeHolderS();
    holder1->append("A", {3,4});
    holder1->build();
    holder1->fill(static_cast<T>(3.1415f));
    for(size_t i=0; i<3; i++)
	for(size_t j=0; j<4; j++)
	    assert( std::abs(static_cast<double>(holder1->get("A")->val({i,j})) - 3.1415)<0.01 );
    cout<<"ok."<<endl;
};

template<typename T, size_t N>
void testMul(typename IBackendFactory<T>::sPtr factory) {
    cout<<"Проверка операции mul()...";
    auto holder=factory->makeHolderS();
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
    timer.toc();
    cout<<endl;
    cout<<"\t вектор на единичную матрицу "<<N<<"x"<<N<<" ...";
    cout<<" ("<<timer<<") ";
    for(size_t i=0; i<N; i++)
	assert(std::abs(A->raw(i) - A1->raw(i))<1e-3);
    cout<<"ok"<<endl;
    cout<<"\t единичная матрица на вектор "<<N<<"x"<<N<<" ...";
    timer.tic();
    A1->mul(E,A);
    timer.toc();
    cout<<" ("<<timer<<") ";
    for(size_t i=0; i<N; i++)
	assert(std::abs(A->raw(i) - A1->raw(i))<1e-3);
    cout<<"ok"<<endl;
    
}

template<typename T>
void testOptGrad(typename IBackendFactory<T>::sPtr factory) {
    cout<<"Производительность градиентного спуска <"<<typeid(T).name()<<">: ";
    auto X=factory->makeHolderS();
    X->append("X", {1000'000});
    X->build();
    auto dX=X->clone();
    Timer timer;
    std::vector<T> regpoly{1e-3,1e-4};
    for(size_t i=0; i<100; i++) {
	dX->get("X")->gaussianNoise(0.0, 1e-3);
	timer.tic();
	X->get("*")->optGrad(dX->get("*"), 1.0, 0.1, regpoly);
	timer.toc();
    };
    cout<<timer<<endl;

};

template<typename T>
void testOptNesterov(typename IBackendFactory<T>::sPtr factory) {
    cout<<"Производительность оптимизатора Нестерова <"<<typeid(T).name()<<">: ";
    auto X=factory->makeHolderS();
    X->append("X", {1000'000});
    X->build();
    auto dX=X->clone();
    auto V=X->clone();
    V->fill();
    Timer timer;
    std::vector<T> regpoly{1e-3,1e-4};
    for(size_t i=0; i<100; i++) {
	dX->get("X")->gaussianNoise(0.0, 1e-3);
	timer.tic();
	X->get("*")->optNesterov(dX->get("*"), V->get("*"), 1.0, 0.1, 0.5,  regpoly);
	timer.toc();
    };
    cout<<timer<<endl;

};




template<typename T>
void testAll(typename IBackendFactory<T>::sPtr factory) {
    testGet<T>(factory);
    testClone<T>(factory);
    testExceptions<T>(factory);
    testFill<T>(factory);
    testMul<T, 10>(factory);
    testMul<T, 100>(factory);
    testMul<T, 1000>(factory);
    testOptGrad<T>(factory);
    testOptNesterov<T>(factory);
};

template<typename T>
void test() {
    cout<<Console().fgColor(Console::aqua).blink()<<"Тест бэкенда OpenMP:"<<endl<<Console().clear();
    auto factory=BackendOpenMP<T>::build();
    testAll<T>(factory);
};



int main()
{
    test<float>();
    test<double>();
    test<long double>();
    return 0;
};

