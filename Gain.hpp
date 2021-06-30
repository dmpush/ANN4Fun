#ifndef __GAIN_HPP_
#define __GAIN_HPP_

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
class Gain : public Learnable<T> {
public:
    Gain() = delete;
    Gain(const Gain&) = delete;
    Gain(ANN<T> *ann) : Learnable<T>(ann) {
	auto shape=ann->getOutputs()->dims();
	Learnable<T>::getParams()->append("K", shape );
	Learnable<T>::getParams()->build();

	Learnable<T>::getGrad()->clone( Learnable<T>::getParams() );
	Learnable<T>::setTutor( std::make_unique<SimpleTutor<T>>() );
	// определяем прямые ссылки на тензоры
	K_=Learnable<T>::getParams()->get("K");
	dK_=Learnable<T>::getGrad()->get("K");

	X_=Learnable<T>::getInputs();
	Y_=Learnable<T>::getOutputs();

	dX_=Learnable<T>::getInputErrors();
	dY_=Learnable<T>::getOutputErrors();

//	Learnable<T>::getParams()->fill(1.0);
	K_->gaussianNoise(0.0, 0.1);
    };


    void forward() override {
	// интерпретируем тензор как одномерный
	for(size_t i=0; i<X_->size(); i++)
	    Y_->raw(i) = X_->raw(i) * K_->raw(i);
    };

    void backward() override {
	for(size_t i=0; i<X_->size(); i++) {
	    // ошибка по входам
	    dX_->raw(i) = K_->raw(i) * dY_->raw(i);
	    // градиент
	    dK_->raw(i) += X_->raw(i) * dY_->raw(i);
	};
	Learnable<T>::backward();
    };
    void dump() override {
	std::cout<<"Gain:"<<std::endl;
	Learnable<T>::getParams()->dump();
	Learnable<T>::getGrad()->dump();
	Successor<T>::dump();
    };


private:
    // ссылки на тензоры для быстрого доступа
    TensorPtr<T> K_, X_, Y_;
    TensorPtr<T> dK_, dX_, dY_;
};

#endif