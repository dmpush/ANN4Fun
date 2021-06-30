#ifndef __LAYER_HPP_
#define __LAYER_HPP_

#include <cmath>
#include <vector>
#include <memory>
#include <stdexcept>
#include <iostream>

#include <Learnable.hpp>
#include <DataHolder.hpp>
#include <AbstractTutor.hpp>
#include <SimpleTutor.hpp>
#include <ANN.hpp>
#include <Tensor.hpp>

/**
    @brief Layer - классический слой нейронной сети - получает на вход вектор, умножает его на матрицу весов,
    и к полученном произведению прибавляет вектор смещений. Вход и выход сети - векторы.
*/
template <typename T>
class Layer : public Learnable<T> {
public:
    Layer() = delete;
    Layer(const Layer&) = delete;
    Layer(ANN<T> *ann, const std::vector<size_t>& Nout) : Learnable<T>(ann, Nout) {
	if(Nout.size() != 1)
	    throw std::runtime_error("Выходы должны быть организованны в 1-тензор");
	if(ann->getOutputs()->dim() != 1)
	    throw std::runtime_error("Входы должны быть организованны в 1-тензор");
	size_t Nin=ann->getOutputs()->size();
	Learnable<T>::getParams()->append("W", {Nout[0], Nin});
	Learnable<T>::getParams()->append("C", {Nout[0]});
	Learnable<T>::getParams()->build();

	Learnable<T>::getGrad()->clone( Learnable<T>::getParams() );
	Learnable<T>::setTutor( std::make_unique<SimpleTutor<T>>() );
	// определяем прямые ссылки на тензоры
	W_=Learnable<T>::getParams()->get("W");
	C_=Learnable<T>::getParams()->get("C");

	dW_=Learnable<T>::getGrad()->get("W");
	dC_=Learnable<T>::getGrad()->get("C");

	X_=Learnable<T>::getInputs();
	Y_=Learnable<T>::getOutputs();

	dX_=Learnable<T>::getInputErrors();
	dY_=Learnable<T>::getOutputErrors();

//	Learnable<T>::getParams()->fill(0.1);
	double S=2.0/static_cast<double>(Learnable<T>::getNumInputs());
//	W_->gaussianNoise<T>(0, std::sqrt(S));
	W_->gaussianNoise(0, std::sqrt(S/2.0));
	C_->gaussianNoise(0, 1e-9);
    };


    void forward() override {
	Y_->mul(X_, W_);
	Y_->append(C_);
    };

    void backward() override {
	// ошибки по входам
	dX_->mul(W_, dY_);
	// градиент синаптической матрицы - внешнее произведение входов и ошибок по выходам
	dW_->extmulapp(dY_, X_);
	// градиент смещений нейронов
	dC_->append( dY_);
	Learnable<T>::backward();
    };
    void dump() override {
	std::cout<<"Layer:"<<std::endl;
	Learnable<T>::getParams()->dump();
	Learnable<T>::getGrad()->dump();
	Successor<T>::dump();
    };


private:
    // ссылки на тензоры для быстрого доступа
    TensorPtr<T> W_, C_, X_, Y_;
    TensorPtr<T> dW_, dC_, dX_, dY_;
};

#endif