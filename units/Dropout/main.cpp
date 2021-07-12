#include <iostream>
#include <string>
#include <cassert>
#include <stdexcept>
#include <memory>
#include <vector>
#include <cstdlib>
#include <cmath>

#include <Dropout.hpp>
#include <Model.hpp>
#include <DataHolder.hpp>
#include <OnlineStatistics.hpp>
using namespace std;

template<typename T>
void exceptMsg(std::runtime_error e) {
	cout<<"    Перехвачено<"<<typeid(T).name()<<">: \""<<e.what()<<"\" ok."<<endl;
};


template<typename T>
void test1() {
    cout<<"\tПроверка статистики на нормальном распределении...";
    bool hasException{false};
    hasException=false;
    try {
	std::vector<size_t> inputShape={100,100};

	auto model=std::make_unique<Model<T>>(inputShape);
	model->template addLayer<Dropout<T>>(0.1);
	typename Dropout<T>::Enabled dropoutOn(true);
	typename Dropout<T>::Enabled dropoutOff(false);
	typename Dropout<T>::Update dropoutUpdate;
	assert(model->getNumInputs() == model->getNumOutputs());
	auto inputs=model->getInputs();
	auto outputs=model->getOutputs();
	auto inputErrors=model->getInputErrors();
	auto outputErrors=model->getOutputErrors();
	OnlineStatistics statFwd, statBwd;
	OnlineStatistics statFwdMean, statBwdMean;
	OnlineStatistics statFwdStd, statBwdStd;
	model->notify(&dropoutOn);
	model->batchBegin();
	size_t batchSize=1000;
	for(size_t smp=0; smp<batchSize; smp++) {
	    model->notify(&dropoutUpdate);
	    inputs->gaussianNoise(0.0, 1.0);
	    model->forward();
	    statFwd.cleanup();
	    for(size_t o=0; o<outputs->size(); o++) {
		auto y=outputs->raw(o);
		statFwd.update(y);
	    };
	    outputs->fill(1.0);
	    outputErrors->gaussianNoise(0.0, 1.0);
	    model->backward();
	    statBwd.cleanup();
	    for(size_t i=0; i<inputs->size(); i++) {
		auto y=inputErrors->raw(i);
		statBwd.update(y);
	    };
	    statFwdMean.update(statFwd.getMean());
	    statFwdStd.update(statFwd.getStd());
	    statBwdMean.update(statBwd.getMean());
	    statBwdStd.update(statBwd.getStd());
	};
	model->batchEnd();
//	cout<<"----------------"<<endl;
//	cout<<statFwdMean.getMean()<<"  "<<statFwdStd.getMean();
//	cout<<"\t";
//	cout<<statBwdMean.getMean()<<"  "<<statBwdStd.getMean();
//	cout<<endl;
	assert(std::abs(statFwdMean.getMean() - 0.0f)<0.01f);
	assert(std::abs(statFwdStd.getMean() - 1.054f)<0.01f);
	assert(std::abs(statBwdMean.getMean() - 0.0f)<0.01f);
	assert(std::abs(statBwdStd.getMean() - 1.054f)<0.01f);
    } catch(std::runtime_error e) {
	exceptMsg<T>(e);
	hasException=true;
    };
    assert( ! hasException);
    cout<<"ok."<<endl;
};





template<typename T>
void test() {
    cout<<"Проверка Dropout<"<<typeid(T).name()<<">..."<<endl;
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

