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

using namespace std;

template<typename T, typename TAct>
auto  getDecoder1() {
    std::vector<size_t> inputShape={10u};
    std::vector<size_t> outputShape={28u,28u};
    auto model = std::make_shared<Model<T>>(inputShape);

    model->template addLayer<Layer<T>>({25});
    model->template addLayer<Dropout<T>>(0.1);
    model->template addLayer<TAct>();

    model->template addLayer<Layer<T>>({49});
    model->template addLayer<Dropout<T>>(0.1);
    model->template addLayer<TAct>();

    model->template addLayer<Layer<T>>({98});
    model->template addLayer<Dropout<T>>(0.1);
    model->template addLayer<TAct>();

    model->template addLayer<Layer<T>>({196});
    model->template addLayer<Dropout<T>>(0.1);
    model->template addLayer<TAct>();

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
auto  getEncoder1() {
    std::vector<size_t> shape={28u,28u};
    auto model = std::make_shared<Model<T>> (shape);
    model->template addLayer<Reshape<T>>({28*28});
//    model->template addLayer<Layer<T>>({784});
//    model->template addLayer<TAct>();
//    model->template addLayer<Layer<T>>({392});
//    model->template addLayer<TAct>();

    model->template addLayer<Layer<T>>({196});
    model->template addLayer<Dropout<T>>(0.1);
    model->template addLayer<TAct>();

    model->template addLayer<Layer<T>>({98});
    model->template addLayer<Dropout<T>>(0.1);
    model->template addLayer<TAct>();

    model->template addLayer<Layer<T>>({49});
    model->template addLayer<Dropout<T>>(0.1);
    model->template addLayer<TAct>();

    model->template addLayer<Layer<T>>({25});
    model->template addLayer<Dropout<T>>(0.1);
    model->template addLayer<TAct>();

    model->template addLayer<Layer<T>>({10});
//    model->template addLayer<Arctan<T>>(); 
    model->template addLayer<Normalize<T>>();  
//    model->template addLayer<SoftMax<T>>();  // вход софтмакса должен быть ограничен
//    model->template setTutor<NesterovTutor<T>>(0.1, 0.5);
    model->build(BackendOpenMP<T>::build());
    model->template setTutor<AdamTutor<T>>();

    return model;
};


template<typename T>
auto  getModel(typename Model<T>::sPtr encoder, typename Model<T>::sPtr decoder) {
    return std::make_shared<Composition<T>>(encoder, decoder);
};

template<typename T>
auto  getGenerator(typename Model<T>::sPtr decoder) {
    auto generator=std::make_shared<Generator<T>>(10u);
    generator->build(BackendOpenMP<T>::build());
    return std::make_shared<Composition<T>>(generator, decoder);
};


template<typename R, typename T>
void test(std::shared_ptr<typename MNIST<R>::MNIST_set>  mnistSet, typename Composition<T>::sPtr model) {
    typename Dropout<T>::Enabled dropout_off(false);
    T trainError{0.0};
    model->notify(&dropout_off);
    for(auto it=mnistSet->begin(); it!=mnistSet->end(); it++) {
	for(size_t q=0; q<28*28; q++)
	    model->setInput(q, (*it)->raw(q));
	model->forward();

	for(size_t q=0; q<28*28; q++) {
	    T delta=model->getOutput(q) - model->getInput(q);
	    trainError+=delta*delta;
	};
    };
    std::cout<<"Error="<<(double)trainError/(double)(mnistSet->numSamples()*28*28) << std::endl;
};

template<typename R, typename T>
void train(std::string prefix, std::deque<typename MNIST<R>::Image::sPtr> dataset, typename ANN<T>::sPtr model, size_t batchSize=10) {
    typename Dropout<T>::Enabled dropout_on(true);
    typename Dropout<T>::Update dropout_update;
    Timer timerFwd,timerBwd,timerTrain;
    model->notify(&dropout_on);
    size_t smp{0};
    T batchError{0};
    for(auto it: dataset) { // начало батча
	if( (smp%batchSize) == 0) {
	    model->notify(&dropout_update);
	    model->batchBegin();
	    batchError=0.0;
	};

	for(size_t q=0; q<28*28; q++)
	    model->setInput(q, it->raw(q));
	timerFwd.tic();
	model->forward();
	timerFwd.toc();
	for(size_t q=0; q<28*28; q++) {
            T delta=model->getOutput(q) - model->getInput(q);
            batchError+=delta*delta;
	    model->setOutput(q, model->getInput(q));
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
	    float err=static_cast<float>(batchError)/static_cast<float>(batchSize*28*28);
	    std::cout<<prefix<<" [elapsed =" <<done<<"%] error="<<err<<std::endl;;
	};
	smp++;
    };
    std::cout<<std::endl;
    std::cout<<"Производительность forward() "<<timerFwd<<endl;
    std::cout<<"Производительность backward() "<<timerBwd<<endl;
    std::cout<<"Производительность batchEnd() "<<timerTrain<<endl;
};

template<typename T>
void saveImages(typename Model<T>::sPtr decoder, size_t ep) {
    auto gen=getGenerator<double>(decoder);
//    auto gen=decoder;
    auto X=gen->getInputs();
    auto Y=gen->getOutputs();
    typename Dropout<T>::Enabled dropout_off(false);
    gen->notify(&dropout_off);
    for(size_t it=0; it<10; it++) {
//	X->fill(0);
//	X->raw(it)=1;
	gen->forward();
	std::string fname="/tmp/gen-ep="+std::to_string(ep)+"-n="+std::to_string(it)+".jpg";
	tensormath::toJPEG<T>(Y, fname);
    };
};

template<typename R, typename T>
void saveImagePairs(std::deque<typename MNIST<R>::Image::sPtr> dataset, typename Composition<T>::sPtr model) {
    T acc[10][10];
    size_t cnt[10]={0};
    for(size_t i=0; i<10; i++) {
	for(size_t j=0; j<10; j++)
	    acc[i][j] = 0.0;
	cnt[i]=0;
    };
    // ищем центроиды значений энкодера для разных классов
    for(auto it: dataset) {
	for(size_t q=0; q<28*28; q++)
	    model->getFirst()->setInput(q, it->raw(q));
	model->getFirst()->forward();
	for(size_t q=0; q<10; q++)
	    acc[it->label()][q] += model->getFirst()->getOutput(q);
	cnt[it->label()]++;
    };
    for(size_t i=0; i<10; i++)  {
	for(size_t j=0; j<10; j++) {
	    acc[i][j]/=static_cast<T>(cnt[i]);
	    std::cout<<acc[i][j]<<"  ";
	}
	std::cout<<std::endl;
    }

    std::ofstream file("/tmp/table.html");
    if(!file.is_open()) {
	cout<<"Не могу создать файл!"<<endl;
    };
    file<<"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 3.2//EN\">"<<endl;
    file<<"<HTML>"<<endl;
    file<<"<meta http-equiv=\"Content-Type\" content=\"text/html; charset=UTF-8\">"<<endl;
    file<<"<HEAD>"<<endl;
    file<<"<TITLE> Отчет </TITLE>"<<endl;
    file<<"</HEAD>"<<endl;
    file<<"<BODY>"<<endl;
    file<<"<H1> Образы</H1><BR>"<<endl;
    file<<"<TABLE border=\"1\">"<<endl;
    // предъявляем декодеру попарные средние
    for(size_t i=0; i<10; i++) {
	file<<"<TR>"<<endl;
	for(size_t j=0; j<10; j++) {
	    for(size_t q=0; q<10; q++)
		model->getSecond()->setInput(q, acc[i][q]*0.33333+acc[j][q]*0.77778 );
	    model->getSecond()->forward();
	    auto Y=model->getSecond()->getOutputs();
	    std::string fname="/tmp/lab="+std::to_string(i)+"x"+std::to_string(j)+".jpg";
	    tensormath::toJPEG<T>(Y, fname);
	    file<<"\t<TD>"<<endl;
	    file<<"\t<IMG src=\""<<fname<<"\"/>"<<endl;
	    file<<"\t</TD>"<<endl;
	};
	file<<"</TR>"<<endl;
    };
    file<<"</TABLE>"<<endl;
    file<<"</BODY>"<<endl;
    file<<"</HTML>"<<endl;

};




int main()
{
    auto mnist=std::make_shared<MNIST<float>>("../../../../");
    auto  encoder=getEncoder1<double, SiLU<double>>();
    auto  decoder=getDecoder1<double, SiLU<double>>();
    auto  autoencoder=getModel<double>(encoder,decoder);
    for(size_t ep=0; ep<100; ep++) {

	auto dataset=mnist->getTrainSet()->shuffle();
	saveImagePairs<float,double>(dataset,autoencoder);
	train<float,double>("epoch #"+std::to_string(ep), dataset, autoencoder);
	cout<<"Ошибка по обучающей выборке:"<<endl;
	test<float,double>(mnist->getTrainSet(), autoencoder);
	cout<<"Ошибка по тестовой выборке:"<<endl;
	test<float,double>(mnist->getTestSet(), autoencoder); 
//	saveImages<double>(decoder, ep);
    };

    return 0;
};
