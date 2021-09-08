#ifndef __GAIN_HPP_
#define __GAIN_HPP_

#include <vector>
#include <memory>
#include <stdexcept>
#include <iostream>

#include <Successor.hpp>
#include <Learnable.hpp>
#include <IBackendFactory.hpp>
#include <AbstractTutor.hpp>
#include <SimpleTutor.hpp>
#include <ANN.hpp>
#include <ITensor.hpp>

/**
    @brief Gain - синаптические связи - усилители. Размерность выходного тензора равна размерности входного тензора.

*/
template <typename T>
class Gain : public Learnable<T> {
public:
    Gain() = delete;
    Gain(const Gain&) = delete;
    Gain(typename ANN<T>::sPtr ann) :
	Learnable<T>(ann, ann->shape()),
	K_{nullptr},
	X_{nullptr},
	Y_{nullptr},
	dK_{nullptr},
	dX_{nullptr},
	dY_{nullptr} {
    };
    void build(typename IBackendFactory<T>::sPtr factory) override {
	Learnable<T>::build(factory);
	Learnable<T>::getParams()->append("K", Successor<T>::shape() );
	Learnable<T>::getParams()->build();
	/// @todo теоретически, здесь может быть проблема, сделать как в Layer
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
	assert(X_);
	assert(Y_);
	assert(K_);
	Y_->prod(X_, K_);
    };

    void backward() override {
	assert(dX_);
	assert(dY_);
	assert(K_);
	assert(dK_);
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