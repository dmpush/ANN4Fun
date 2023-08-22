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
#include <ReLU.hpp>
#include <SoftMax.hpp>
#include <Arctan.hpp>
#include <SELU.hpp>
#include <SiLU.hpp>
#include <Dropout.hpp>
#include <AdamTutor.hpp>
#include <NesterovTutor.hpp>
#include <Timer.hpp>
using namespace std;

template<
    typename T,
    template<typename>
	class Act
>
auto  getModel1() {
    std::vector<size_t> shape={28u,28u};
    auto model = std::make_shared<Model<T>> (shape);
    model->template addLayer<Reshape<T>>({28*28});
//    model->template addLayer<Layer<T>>({784});
//    model->template addLayer<ReLU<T>>();
//    model->template addLayer<Layer<T>>({392});
//    model->template addLayer<ReLU<T>>();

    model->template addLayer<Layer<T>>({196});
    model->template addLayer<Dropout<T>>(0.1);
    model->template addLayer<Act<T>>();

    model->template addLayer<Layer<T>>({98});
    model->template addLayer<Dropout<T>>(0.1);
    model->template addLayer<Act<T>>();

    model->template addLayer<Layer<T>>({49});
    model->template addLayer<Dropout<T>>(0.1);
    model->template addLayer<Act<T>>();

    model->template addLayer<Layer<T>>({25});
    model->template addLayer<Dropout<T>>(0.1);
    model->template addLayer<Act<T>>();

    model->template addLayer<Layer<T>>({10});
//    model->template addLayer<Arctan<T>>(); // вход софтмакса должен быть ограничен
    model->template addLayer<SoftMax<T>>();
    model->build(BackendOpenMP<T>::build());
//    model->template setTutor<NesterovTutor<T>>(0.1, 0.5);
    model->template setTutor<AdamTutor<T>>();
    return model;
};


template<typename R, typename T>
void test(std::shared_ptr<typename MNIST<R>::MNIST_set>  mnistSet, typename Model<T>::sPtr model) {
    typename Dropout<T>::Enabled dropout_off(false);
    size_t errors_count=0;
    model->notify(&dropout_off);
    for(auto it=mnistSet->begin(); it!=mnistSet->end(); it++) {
	for(size_t q=0; q<28*28; q++)
	    model->setInput(q, (*it)->raw(q));
	model->forward();

	size_t lab=0;
	for(size_t q=1; q<10; q++) 
	    if(model->getOutput(q) >= model->getOutput(lab))
		lab=q;
	if(lab!=(*it)->label())
		errors_count++;
    };
    std::cout<<"Error="<<errors_count<<" from "
	<<mnistSet->numSamples()
	<<" ("
	<<100.0*(double)errors_count/(double)mnistSet->numSamples()
	<<"%)"
	<<std::endl;
};

template<typename R, typename T>
auto train(std::string prefix, std::deque<typename MNIST<R>::Image::sPtr> dataset, typename Model<T>::sPtr model,  size_t batchSize=10) {
    std::deque<typename MNIST<R>::Image::sPtr> wrong;
    typename Dropout<T>::Enabled dropout_on(true);
    typename Dropout<T>::Update dropout_update;
    Timer timerFwd,timerBwd,timerTrain;
    model->notify(&dropout_on);
    size_t smp=0;
	size_t batchError{0};
    for(auto it: dataset) { // начало батча
	if( (smp%batchSize) == 0) {
	    model->notify(&dropout_update);
	    model->batchBegin();
	    batchError=0;
	};

	for(size_t q=0; q<28*28; q++)
	    model->setInput(q, it->raw(q));
	timerFwd.tic();
	model->forward();
	timerFwd.toc();
	size_t lab=0;
	for(size_t q=1; q<10; q++) 
	    if(model->getOutput(q) >= model->getOutput(lab))
		lab=q;
	if(lab!=it->label()) {
	    wrong.push_back(it);
	    batchError++;
	};
	for(size_t q=0; q<10; q++) {
	    T target=it->label()==q ? 1.0 : 0.0;
	    model->setOutput(q, target);
	};
	timerBwd.tic();
	model->backward();
	timerBwd.toc();
	// конец батча или эпохи
	if( (smp%batchSize) == batchSize-1 || smp+1==dataset.size()) { 
	    timerTrain.tic();
	    model->batchEnd();
	    timerTrain.toc();
	    float done=static_cast<float>(smp*100)/static_cast<float>(dataset.size());
	    float err=static_cast<float>(batchError*100)/static_cast<float>(batchSize);
	    std::cout<<prefix<<" [elapsed =" <<done<<"%] error="<<err<<"%"<<std::endl;;
	};
	smp++;
    };
    std::cout<<std::endl;
    std::cout<<"Производительность forward() "<<timerFwd<<endl;
    std::cout<<"Производительность backward() "<<timerBwd<<endl;
    std::cout<<"Производительность batchEnd() "<<timerTrain<<endl;
    return wrong;
};

int main()
{
    auto mnist=std::make_shared<MNIST<float>>("../../../../");
    auto model=getModel1<double, SiLU>();
    for(size_t ep=0; ep<1000; ep++) {
	auto dataset=mnist->getTrainSet()->shuffle();
	auto wrong=train<float,double>("epoch #"+std::to_string(ep), dataset, model);
	size_t cnt=0;
        while(wrong.size()>0 && cnt<1
) {
	    mnist->shuffle(wrong);
	    wrong=train<float,double>("mistakes correction "+std::to_string(ep)+"."+std::to_string(cnt), wrong, model);
	    cnt++;
        };
	cout<<"Ошибка по обучающей выборке:"<<endl;
	test<float,double>(mnist->getTrainSet(), model);
	cout<<"Ошибка по тестовой выборке:"<<endl;
	test<float,double>(mnist->getTestSet(), model);
    };

    return 0;
};

