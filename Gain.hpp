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
    @brief Gain - синаптические связи - усилители. Размерность выходного тензора равна размерности входного тензора.

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

	K_->gaussianNoise(0.0, 0.1);
    };


    void forward() override {
	Y_->prod(X_, K_);
    };

    void backward() override {
	// ошибка по входам
	dX_->prod(K_, dY_);
	// градиент весов
	dK_->prodapp(X_, dY_);
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