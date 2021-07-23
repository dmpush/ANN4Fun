#include <iostream>
#include <string>
#include <cassert>
#include <stdexcept>
#include <memory>
#include <vector>
#include <cmath>

#include <Input.hpp>
#include <Layer.hpp>
#include <ANN.hpp>
#include <TestRot3.hpp>
#include <NesterovTutor.hpp>
#include <BackendOpenMP.hpp>
using namespace std;

template<typename T>
void exceptMsg(std::runtime_error e) {
	cout<<"    Перехвачено<"<<typeid(T).name()<<">: \""<<e.what()<<"\" ok."<<endl;
};




template<typename T>
class NesterovRot3 : public TestRot3<T> {
    T dt_, beta_;
public:
    NesterovRot3(T dt, T beta): TestRot3<T>(), dt_(dt), beta_(beta) { };

    std::shared_ptr<Model<T>> buildModel() override {
	std::vector<size_t> inputShape{TestRot3<T>::getNumInputs()};
	std::vector<size_t> outputShape{TestRot3<T>::getNumOutputs()};
	auto model=std::make_shared<Model<T>> ( inputShape );
	model-> template addLayer<Layer<T>>(outputShape);
//	model-> template setTutor<NesterovTutor<T>>(dt_,beta_);
	typename AbstractTutor<T>::uPtr tut=std::make_unique<NesterovTutor<T>>(dt_, beta_);
	model->build(BackendOpenMP<T>::build());
	model->setTutor(std::move(tut));
	return model;
    };
    bool assertion() override {
	return TestRot3<T>::getErrorMeanSquare()<0.001;
    };
};


template<typename T>
void testNesterov(T dt, T beta) {
    cout<<"Проверка метода оптимизации Nesterov<"<<typeid(T).name()<<">("<<dt<<","<<beta<<") на 3D операторе вращения ...";
    NesterovRot3<T> test(dt, beta);
    auto cnt=test.run(1000);
    assert(cnt>10);
    cout<<"ok."<<endl;

};


template<typename T>
void test() {
    testNesterov<T>(0.1, 0.9);
    testNesterov<T>(0.1, 0.5);
    testNesterov<T>(0.1, 0.1);
};



int main()
{
    test<float>();
    test<double>();
    test<long double>();
    return 0;
};


