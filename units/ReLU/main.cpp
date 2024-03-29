#include <iostream>
#include <string>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <vector>
#include <memory>
#include <functional>

#include <BackendOpenMP.hpp>
#include <SimpleTutor.hpp>
#include <Input.hpp>
#include <Layer.hpp>
#include <Model.hpp>
#include <ReLU.hpp>
#include <Gain.hpp>

#include <Assertion.hpp>
#include <TestXOR.hpp>
#include <TestGains.hpp>

template<typename T>
class ReLUxor : public TestXOR<T> {
public:
    std::shared_ptr<Model<T>> buildModel() override {
	std::vector<size_t> inputShape{TestXOR<T>::getNumInputs()};
	std::vector<size_t> outputShape{TestXOR<T>::getNumOutputs()};
	std::function<void(T)> validValue=[](T x) { 
	    assert(!std::isnan(x));
	    assert(!std::isinf(x));
	};
	auto model=std::make_shared<Model<T>> ( inputShape );
	model-> template addLayer<Layer<T>>({7});
	model-> template addLayer<Assertion<T>>(validValue, validValue);
	model-> template addLayer<ReLU<T>>();
	model-> template addLayer<Assertion<T>>(validValue, validValue);
	model-> template addLayer<Layer<T>>({5});
	model-> template addLayer<Assertion<T>>(validValue, validValue);
	model-> template addLayer<ReLU<T>>();
	model-> template addLayer<Assertion<T>>(validValue, validValue);
	model-> template addLayer<Layer<T>>( outputShape );
	model-> template addLayer<Assertion<T>>(validValue, validValue);
	model->build(BackendOpenMP<T>::build());
	return model;
    };
    bool assertion() override {
	return TestXOR<T>::getErrorMeanSquare()<0.1;
    };
};

template<typename T>
class ReLUGains: public TestGains<T> {
public:
    std::shared_ptr<Model<T>> buildModel() override {
	std::vector<size_t> inputShape{TestGains<T>::getNumInputs()};
	std::vector<size_t> outputShape{TestGains<T>::getNumOutputs()};
	std::function<void(T)> validValue=[](T x) { 
	    assert(!std::isnan(x));
	    assert(!std::isinf(x));
	};
	auto model=std::make_shared<Model<T>> (inputShape);
	model-> template addLayer<Layer<T>>({3});
	model-> template addLayer<Assertion<T>>(validValue, validValue);
	model-> template addLayer<ReLU<T>>();
	model-> template addLayer<Assertion<T>>(validValue, validValue);
	model-> template addLayer<Layer<T>>({5});
	model-> template addLayer<Assertion<T>>(validValue, validValue);
	model-> template addLayer<ReLU<T>>();
	model-> template addLayer<Assertion<T>>(validValue, validValue);
	model-> template addLayer<Layer<T>>( outputShape );
	model-> template addLayer<Assertion<T>>(validValue, validValue);
	model->build(BackendOpenMP<T>::build());
	return model;
    };
    bool assertion() override {
	return TestGains<T>::getErrorMeanSquare()<0.1;
    };
};



template<typename T>
void test1() {
    cout<<"Проверка ReLU<"<<typeid(T).name()<<"> на логических функциях: ";
    ReLUxor<T> test;
    auto cnt=test.run();
    assert(cnt>=10);
    std::cout<<"ok."<<std::endl;
};

template<typename T>
void test2() {
    cout<<"Проверка ReLU<"<<typeid(T).name()<<"> на усилителях: ";
    ReLUGains<T> test;
    auto cnt=test.run();
    assert(cnt>=10);
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

