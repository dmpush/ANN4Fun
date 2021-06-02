#ifndef __LAYER_HPP_
#define __LAYER_HPP_

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
    Layer(ANN<T> *ann, size_t Nout) : Learnable<T>(ann, Nout) {
	params_=std::make_shared<DataHolder<T>>();
	if(ann->getOutputs()->dim() != 1)
	    throw std::runtime_error("Входная сеть иметь выход 1-тензор");
	params_->append("W", {Nout, ann->getOutputs()->size()});
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
	tensormath::copy<T>(params_->get("C"), Successor<T>::getOutputs());
	tensormath::mul<T>(Successor<T>::getInputs(), params_->get("W"), Successor<T>::getOutputs());
    };
    void backward() override {
	// ошибки по входам
	tensormath::mul<T>(params_->get("W"), Successor<T>::getOutputErrors(), Successor<T>::getInputErrors());
	// градиент синаптической матрицы - внешнее произведение входов и ошибок по выходам
	tensormath::extmulapp<T>(Successor<T>::getOutputErrors(),Successor<T>::getInputs(), grad_->get("W"));
	// градиент смещений нейронов
	tensormath::append<T>(grad_->get("C"), Successor<T>::getOutputErrors());
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