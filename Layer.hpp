#ifndef __LAYER_HPP_
#define __LAYER_HPP_

#include <vector>
#include <memory>
#include <stdexcept>

#include <Learnable.hpp>
#include <DataHolder.hpp>
#include <AbstractTutor.hpp>
#include <SimpleTutor.hpp>
#include <ANN.hpp>
#include <TensorMath.hpp>

template <typename T>
class Layer : public Learnable<T> {
public:
    Layer() = delete;
    Layer(const Layer&) = delete;
    Layer(ANN<T> *ann, std::vector<size_t> Nout) : Learnable<T>(ann, Nout) {
	if(Nout.size() != 1)
	    throw std::runtime_error("Выходы должны быть организованны в 1-тензор");
	if(ann->getOutputs()->dim() != 1)
	    throw std::runtime_error("Входы должны быть организованны в 1-тензор");
	size_t Nin=ann->getOutputs()->size();
	Learnable<T>::getParams()->append("W", {Nout[0], Nin});
	Learnable<T>::getParams()->append("C", {Nout[0]});
	Learnable<T>::getParams()->build();
	Learnable<T>::getParams()->fill(0.1);

	Learnable<T>::getGrad()->clone( Learnable<T>::getParams() );
//	Learnable<T>::getParams()->description();
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
    };


    void forward() override {
	tensormath::mul<T>(X_, W_, Y_);
	tensormath::append<T>(C_, Y_);
    };

    void backward() override {
	// ошибки по входам
	tensormath::mul<T>(W_, dY_, dX_);
	// градиент синаптической матрицы - внешнее произведение входов и ошибок по выходам
	tensormath::extmulapp<T>(dY_, X_, dW_);
	// градиент смещений нейронов
	tensormath::copy<T>( dY_, dC_);
	Learnable<T>::backward();
    };



private:
    // ссылки на тензоры для быстрого доступа
    Tensor<T> W_, C_, X_, Y_;
    Tensor<T> dW_, dC_, dX_, dY_;
};

#endif