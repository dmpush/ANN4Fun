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
	params_->description();
	grad_->description();
	setupTutor( std::make_unique<SimpleTutor<T>>() );
    };
    void setupTutor(typename AbstractTutor<T>::uPtr tutor) override {
	Learnable<T>::setTutor(std::move(tutor));
	Learnable<T>::setContext(params_, grad_); // чойтакактанитак
    };


    void forward() override {
	tensormath::copy<T>(params_->get("C"), Learnable<T>::getOutputs());
	tensormath::mul<T>(Learnable<T>::getInputs(), params_->get("W"), Learnable<T>::getOutputs());
    };
    void backward() override {
	// ошибки по входам
	tensormath::mul<T>(params_->get("W"), Learnable<T>::getOutputErrors(), Learnable<T>::getInputErrors());
	// градиент синаптической матрицы
	tensormath::extmulapp<T>(Learnable<T>::getOutputErrors(), Learnable<T>::getInputs(), grad_->get("W"));
	// градиент смещений нейронов
	tensormath::append<T>(grad_->get("C"), Learnable<T>::getOutputErrors());
	Learnable<T>::backward();
    };

    void batchBegin() override {
	Learnable<T>::batchBegin();
    };

    void batchEnd() override {
	Learnable<T>::batchEnd();
    };


private:
    typename DataHolder<T>::sPtr params_;
    typename DataHolder<T>::sPtr grad_;
};

#endif