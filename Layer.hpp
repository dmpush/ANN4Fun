#ifndef __LAYER_HPP_
#define __LAYER_HPP_

#include <memory>
#include <Learnable.hpp>
#include <DataHolder.hpp>
#include <AbstractTutor.hpp>
#include <SimpleTutor.hpp>
#include <ANN.hpp>
#include <TensorMath.hpp>

template <typename T>
class Layer : public Learnable<T> {
public:
    Layer(size_t Nin, size_t Nout) : Learnable<T>(Nin, Nout) {
	params_=std::make_shared<DataHolder<T>>();
	params_->append("W", {Nout, Nin});
	params_->append("C", {Nout});
	params_->build();
	grad_=params_->clone();
	params_->fill(0.1);
	setTutor(std::make_shared<SimpleTutor<T>>());
	params_->description();
	grad_->description();
    };

    void setTutor(typename AbstractTutor<T>::sPtr tutor) override {
	tutor_=tutor;
	tutor_->setContext(params_, grad_);
    };

    void forward() override {
	tensormath::copy<T>(params_->get("C"), Learnable<T>::Y_);
	tensormath::mul<T>(Learnable<T>::X_, params_->get("W"), Learnable<T>::Y_);
    };
    void backward() override {
	// ошибки по входам
	tensormath::mul<T>(params_->get("W"), Learnable<T>::dY_, Learnable<T>::dX_);
	// градиент синаптической матрицы
	tensormath::extmulapp<T>(Learnable<T>::dY_, Learnable<T>::X_, grad_->get("W"));
	// градиент смещений нейронов
	tensormath::append<T>(grad_->get("C"), Learnable<T>::dY_);
	tutor_->backward();
	Learnable<T>::backward();
    };

    void batchBegin() override {
	Learnable<T>::batchBegin();
	tutor_->batchBegin();
    };

    void batchEnd() override {
	if(ANN<T>::isTrainable())
	    tutor_->batchEnd();
	Learnable<T>::batchEnd();
    };


private:
    typename DataHolder<T>::sPtr params_;
    typename DataHolder<T>::sPtr grad_;
    typename AbstractTutor<T>::sPtr tutor_;
};

#endif