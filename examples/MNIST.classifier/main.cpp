#include <iostream>
#include <string>
#include <cassert>
#include <stdexcept>
#include <memory>
#include <cmath>
#include <vector>

#include <DataHolder.hpp>
#include <MNIST.hpp>
#include <Tensor2JPEG.hpp>
#include <Model.hpp>
#include <Reshape.hpp>
#include <Layer.hpp>
#include <ReLU.hpp>
#include <SoftMax.hpp>
#include <Arctan.hpp>
#include <SELU.hpp>
using namespace std;

template<typename T>
auto  getModel1() {
    std::vector<size_t> shape={28u,28u};
    auto model = std::make_shared<Model<T>> (shape);
    model->template addLayer<Reshape<T>>({28*28});
//    model->template addLayer<Layer<T>>({784});
//    model->template addLayer<ReLU<T>>();
//    model->template addLayer<Layer<T>>({392});
//    model->template addLayer<ReLU<T>>();
    model->template addLayer<Layer<T>>({196});
    model->template addLayer<ReLU<T>>();
    model->template addLayer<Layer<T>>({98});
    model->template addLayer<ReLU<T>>();
    model->template addLayer<Layer<T>>({49});
    model->template addLayer<ReLU<T>>();
    model->template addLayer<Layer<T>>({25});
    model->template addLayer<ReLU<T>>();
    model->template addLayer<Layer<T>>({10});
//    model->template addLayer<Arctan<T>>();
    model->template addLayer<SoftMax<T>>();
    return model;
};

template<typename R, typename T>
void train(typename MNIST<R>::sPtr mnist, typename Model<T>::sPtr model,  size_t batchSize=10) {
    auto trainSet=mnist->getTrainSet();
    size_t numBatches=trainSet->numSamples()/batchSize;
    for(size_t bat=0; bat<numBatches; bat++) {
	size_t batchError=0;
	model->batchBegin();
	for(size_t smp=0; smp<batchSize; smp++) {
	    auto t=trainSet->getRandomSample();
	    for(size_t q=0; q<28*28; q++)
		model->setInput(q, t->data[q]);
	    model->forward();

	    size_t lab=0;
	    for(size_t q=1; q<10; q++) 
		if(model->getOutput(q) >= model->getOutput(lab))
		    lab=q;
	    if(lab!=t->label)
		batchError++;
	    for(size_t q=0; q<10; q++) {
		T target=t->label==q ? 1.0 : 0.0;
		model->setOutput(q, target);
	    };
	    model->backward();
	};
	model->batchEnd();
	float done=static_cast<float>(bat*100)/static_cast<float>(numBatches);
	float err=static_cast<float>(batchError*100)/static_cast<float>(batchSize);
	std::cout<<"elapsed =" <<done<<"%, error="<<err<<"%"<<std::endl;;
    };
    std::cout<<std::endl;
};

template<typename R, typename T>
void test(typename MNIST<R>::sPtr mnist, typename Model<T>::sPtr model) {
    auto testSet=mnist->getTestSet();
    size_t errors_count=0;
    for(auto it=testSet->begin(); it!=testSet->end(); it++) {
	for(size_t q=0; q<28*28; q++)
	    model->setInput(q, (*it)->data[q]);
	model->forward();

	size_t lab=0;
	for(size_t q=1; q<10; q++) 
	    if(model->getOutput(q) >= model->getOutput(lab))
		lab=q;
	if(lab!=(*it)->label)
		errors_count++;
	};
    std::cout<<"Error="<<errors_count<<" from "
	<<testSet->numSamples()
	<<" ("
	<<100.0*(double)errors_count/(double)testSet->numSamples()
	<<"%)"
	<<std::endl;
};


int main()
{
    auto mnist=std::make_shared<MNIST<float>>("../../../");
    auto model=getModel1<double>();
    for(size_t ep=0; ep<10; ep++) {
	train<float,double>(mnist, model);
	test<float,double>(mnist, model);
    };

    return 0;
};

