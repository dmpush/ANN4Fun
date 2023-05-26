#include <iostream>
#include <string>
#include <cassert>
#include <stdexcept>
#include <memory>
#include <vector>
#include <cmath>

#include <Input.hpp>
#include <PCA_Layer.hpp>
#include <Assertion.hpp>
#include <ANN.hpp>
#include <TestRot3.hpp>
#include <BackendOpenMP.hpp>
using namespace std;

template<typename T>
void exceptMsg(std::runtime_error e) {
	cout<<"    Перехвачено<"<<typeid(T).name()<<">: \""<<e.what()<<"\" ok."<<endl;
};




template<typename T>
class LayerRot3 : public TestRot3<T> {
public:
    std::shared_ptr<Model<T>> buildModel() override {
	std::vector<size_t> inputShape{TestRot3<T>::getNumInputs()};
	std::vector<size_t> outputShape{TestRot3<T>::getNumOutputs()};
	std::function<void(T)> validValue=[](T x) { 
	    assert(!std::isnan(x));
	    assert(!std::isinf(x));
	};
	auto model=std::make_shared<Model<T>> ( inputShape );
	model-> template addLayer<PCA_Layer<T>>(outputShape, 3);
	model-> template addLayer<Assertion<T>>(validValue, validValue);
	model->build(BackendOpenMP<T>::build());
	return model;
    };
    bool assertion() override {
	return TestRot3<T>::getErrorMeanSquare()<0.001;
    };
};


template<typename T>
void testTrain() {
    cout<<"Проверка Layer<"<<typeid(T).name()<<"> обучаемости на 3D операторе вращения ...";
    LayerRot3<T> test;
    auto cnt=test.run(1000);
    assert(cnt>10);
    cout<<"ok."<<endl;

};


template<typename T>
void test() {
    testTrain<T>();
};



int main()
{
    test<float>();
    test<double>();
    test<long double>();
    return 0;
};


