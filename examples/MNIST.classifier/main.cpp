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
    model->template addLayer<Arctan<T>>();
    model->template addLayer<SoftMax<T>>();
    return model;
};


int main()
{
    MNIST<float> mnist("../../../");
    auto trainSet=mnist.getTrainSet();
    auto model=getModel1<double>();

    size_t numEpoches=10;
    size_t batchSize=10;
    size_t numBatches=trainSet->numSamples()/batchSize;
    for(size_t ep=0; ep<10; ep++) {
	std::cout<<"Epoch "<<ep+1<<" from "<< numEpoches<<std::endl;
	for(size_t bat=0; bat<numBatches; bat++) {
	    std::cout<<static_cast<float>(bat*100)/static_cast<float>(numBatches)<<std::endl;
	    float batchError=0.0;
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
		    float target=t->label==q ? 1.0 : 0.0;
		    model->setOutput(q, target);
		};
		model->backward();
	    };
	    model->batchEnd();
	    batchError/=(float)batchSize;
	    std::cout<<"Error="<<batchError*100<<std::endl;
	};
    };

    return 0;
};

