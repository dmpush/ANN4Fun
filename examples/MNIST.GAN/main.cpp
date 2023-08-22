#include <fstream>
#include <iostream>
#include <string>
#include <cassert>
#include <stdexcept>
#include <memory>
#include <cmath>
#include <vector>
#include <deque>

#include <BackendOpenMP.hpp>
#include <MNIST.hpp>
#include <Tensor2JPEG.hpp>
#include <Model.hpp>
#include <Reshape.hpp>
#include <Layer.hpp>
#include <Dropout.hpp>
#include <Input.hpp>
#include <Generator.hpp>
#include <AdamTutor.hpp>
#include <NesterovTutor.hpp>
#include <Composition.hpp>
#include <Timer.hpp>

#include <ReLU.hpp>
#include <ReLUx.hpp>
#include <SoftMax.hpp>
#include <Normalize.hpp>
#include <Arctan.hpp>
#include <SELU.hpp>
#include <SiLU.hpp>
#include <CrossEntropy.hpp>
using namespace std;

template<typename T, typename TAct>
std::shared_ptr<Model<T>>  getGenerator() {
    std::vector<size_t> inputShape={49u};
    std::vector<size_t> outputShape={28u,28u};
    auto model = std::make_shared<Model<T>>(inputShape);

    model->template addLayer<Layer<T>>({64});
    model->template addLayer<Dropout<T>>(0.1);
    model->template addLayer<TAct>();

    model->template addLayer<Layer<T>>({81});
    model->template addLayer<Dropout<T>>(0.1);
    model->template addLayer<TAct>();

//    model->template addLayer<Layer<T>>({98});
//    model->template addLayer<Dropout<T>>(0.1);
//    model->template addLayer<TAct>();

//    model->template addLayer<Layer<T>>({196});
//    model->template addLayer<Dropout<T>>(0.1);
//    model->template addLayer<TAct>();

//    model->template addLayer<Layer<T>>({392});
//    model->template addLayer<TAct>();

    model->template addLayer<Layer<T>>({28*28});
    model->template addLayer<Arctan<T>>();

    model->template addLayer<Reshape<T>>(outputShape);
    model->build(BackendOpenMP<T>::build());
    model->template setTutor<AdamTutor<T>>();
    return model;
};


template<typename T, typename TAct>
auto  getDescriminator() {
    std::vector<size_t> shape={28u,28u};
    auto model = std::make_shared<Model<T>> (shape);
    model->template addLayer<Reshape<T>>({28*28});
//    model->template addLayer<Layer<T>>({784});
//    model->template addLayer<TAct>();
//    model->template addLayer<Layer<T>>({392});
//    model->template addLayer<TAct>();

    model->template addLayer<Layer<T>>({81});
    model->template addLayer<Dropout<T>>(0.1);
    model->template addLayer<TAct>();

    model->template addLayer<Layer<T>>({64});
    model->template addLayer<Dropout<T>>(0.1);
    model->template addLayer<TAct>();

//    model->template addLayer<Layer<T>>({49});
//    model->template addLayer<Dropout<T>>(0.1);
//    model->template addLayer<TAct>();

//    model->template addLayer<Layer<T>>({25});
//    model->template addLayer<Dropout<T>>(0.1);
//    model->template addLayer<TAct>();

//    model->template addLayer<Layer<T>>({12});
//    model->template addLayer<Dropout<T>>(0.1);
//    model->template addLayer<TAct>();

//    model->template addLayer<Layer<T>>({6});
//    model->template addLayer<Dropout<T>>(0.1);
//    model->template addLayer<TAct>();

    model->template addLayer<Layer<T>>({2});
    model->template addLayer<SoftMax<T>>();
    model->template addLayer<CrossEntropy<T>>();


    model->build(BackendOpenMP<T>::build());
    model->template setTutor<AdamTutor<T>>();

    return model;
};


template<typename T>
auto  getModel(typename Model<T>::sPtr encoder, typename Model<T>::sPtr decoder) {
    return std::make_shared<Composition<T>>(encoder, decoder);
};


template<typename R, typename T>
void train(std::string prefix, std::shared_ptr<typename MNIST<R>::MNIST_set> dataset, std::shared_ptr<Composition<T>> model,  size_t batchSize=10) {
    auto gen=model->getFirst();
    auto dis=model->getSecond();
    typename Dropout<T>::Enabled dropout_on(true);
    typename Dropout<T>::Update dropout_update;
    T img[28*28];
    std::vector<T> clazz;
    Timer timerFwd,timerBwd,timerTrain;
//    model->notify(&dropout_on);
    size_t smp{0};
    T batchError{0};
    
    size_t numBatches=dataset->numSamples()/batchSize;
    for (size_t batch=0; batch<numBatches; batch++) {
	// обучение дискриминатора на смеси настоящих и фальшивых примеров
	dis->unlockTrain();
	dis->batchBegin();
	batchError=0;
	for(size_t sample=0; sample<batchSize; sample++) {
	    if(rand()%2) { // фальшивый пример
		gen->getInputs()->gaussianNoise(0.0,1.0);
		gen->forward();
		for(size_t q=0; q<28*28; q++)
		    img[q]=gen->getOutput(q);
		clazz={1.0, 0.0};
	    } else {
		auto it=dataset->getRandomSample();
		for(size_t q=0; q<28*28; q++)
		    img[q]= it->raw(q);
		clazz={0.0, 1.0};
	    };
	    for(size_t q=0; q<28*28; q++)
		dis->setInput(q, img[q]);
	    dis->forward();
	    for(size_t q=0; q<2; q++) {
		T errq=clazz[q]-dis->getOutput(q);
		batchError+=errq*errq;
		dis->setError(q, clazz[q]);
//		dis->setOutput(q, clazz[q]);
	    };
	    dis->backward();
	};
	dis->batchEnd();
	//----
	clazz={0.0, 1.0};
	dis->lockTrain();
	gen->batchBegin();
	for(size_t sample=0; sample<batchSize; sample++) {
	    model->getInputs()->gaussianNoise(0.0, 1.0);
	    model->forward();
	    for(size_t q=0; q<2; q++)
		model->setError(q, clazz[q]);
//		model->setOutput(q, clazz[q]);
	    model->backward();
	};
	gen->batchEnd();
	std::cout<<batch<<"/"<<numBatches<<": "<< batchError/(T)batchSize<<std::endl;
    };
};

template<typename T>
void saveImages(typename Model<T>::sPtr gen, size_t ep) {
    auto X=gen->getInputs();
    auto Y=gen->getOutputs();
    typename Dropout<T>::Enabled dropout_off(false);
    gen->notify(&dropout_off);
    for(size_t it=0; it<10; it++) {
	gen->getInputs()->gaussianNoise(0.0, 1.0);
	gen->forward();
	std::string fname="/tmp/gen-ep="+std::to_string(ep)+"-n="+std::to_string(it)+".jpg";
	tensormath::toJPEG<T>(Y, fname);
    };
};

int main(int argc, char *argv[])
{
    auto mnist=std::make_shared<MNIST<float>>("../../../../");
    auto  gen=getGenerator<double, SiLU<double>>();
    auto  dis=getDescriminator<double, Arctan<double>>();
    auto  model=getModel<double>(gen,dis);
    for(size_t ep=0; ep<100; ep++) {

	auto dataset=mnist->getTrainSet()->shuffle();
	train<float,double>("epoch #"+std::to_string(ep), mnist->getTrainSet(), model);
	saveImages<double>(gen, ep);
    };

    return 0;
};
