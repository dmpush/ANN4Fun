#ifndef __LINEAR_HPP_
#define __LINEAR_HPP_

#include <cmath>
#include <vector>
#include <memory>
#include <stdexcept>
#include <iostream>

#include <Learnable.hpp>
#include <IBackendFactory.hpp>
#include <IDataHolder.hpp>
#include <AbstractTutor.hpp>
#include <SimpleTutor.hpp>
#include <ANN.hpp>

/**
    @brief Linear - классический слой нейронной сети - получает на вход вектор, умножает его на матрицу весов,
    и к полученном произведению прибавляет вектор смещений. Вход и выход сети - векторы.
*/
template <typename T>
class Linear : public Learnable<T> {
public:
    using sPtr=std::shared_ptr<Linear<T>>;
    Linear() = delete;
    Linear(const Linear&) = delete;
    Linear(typename ANN<T>::sPtr ann, const std::vector<size_t>& Nout) : Learnable<T>(ann, Nout),
    W_{nullptr},
    X_{nullptr},
    Y_{nullptr},
    dW_{nullptr},
    dX_{nullptr},
    dY_{nullptr} {
    };
    
    void build(typename IBackendFactory<T>::sPtr factory) override {
	Learnable<T>::build(factory);
	auto inputShape=Successor<T>::getPrecursor()->shape();
	auto outputShape=Learnable<T>::getOutputs()->dims();
	if(outputShape.size() != 1)
	    throw std::runtime_error("Выходы должны быть организованны в 1-тензор");
	if(inputShape.size() != 1)
	    throw std::runtime_error("Входы должны быть организованны в 1-тензор");
	Learnable<T>::getParams()->append("W", {outputShape[0], inputShape[0]});
	Learnable<T>::getParams()->build();
	setTutor(std::make_unique<SimpleTutor<T>>() );

	dX_=Learnable<T>::getInputErrors();
	dY_=Learnable<T>::getOutputErrors();

//	Learnable<T>::getParams()->fill(0.1);
	double S=2.0/static_cast<double>(Learnable<T>::getNumInputs());
//	W_->gaussianNoise<T>(0, std::sqrt(S));
	W_->gaussianNoise(0, std::sqrt(S/2.0));
    };
    void setTutor(typename AbstractTutor<T>::uPtr tutor) override {
	Learnable<T>::setTutor( std::move(tutor)  );
	// определяем прямые ссылки на тензоры
	W_=Learnable<T>::getParams()->get("W");

	dW_=Learnable<T>::getGrad()->get("W");

	X_=Learnable<T>::getInputs();
	Y_=Learnable<T>::getOutputs();

	
    }


    void forward() override {
	assert(X_);
	assert(Y_);
	assert(W_);
	Y_->mul(X_, W_);
    };

    void backward() override {
	assert(X_);
	assert(W_);
	assert(dX_);
	assert(dY_);
	assert(dW_);
	// ошибки по входам
	dX_->mul(W_, dY_);
	// градиент синаптической матрицы - внешнее произведение входов и ошибок по выходам
	dW_->extmulapp(dY_, X_);
	Learnable<T>::backward();
    };
    void dump() override {
	std::cout<<"Linear:"<<std::endl;
	Learnable<T>::getParams()->dump();
	Learnable<T>::getGrad()->dump();
	Successor<T>::dump();
    };


private:
    // ссылки на тензоры для быстрого доступа
    TensorPtr<T> W_, X_, Y_;
    TensorPtr<T> dW_, dX_, dY_;
};

#endif