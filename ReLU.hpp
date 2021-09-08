#ifndef __RELU_HPP__
#define __RELU_HPP__

#include <stdexcept>
#include <string>
#include <memory>
#include <vector>

#include <ANN.hpp>
#include <Successor.hpp>
#include <IBackendFactory.hpp>
#include <AbstractTutor.hpp>
#include <ITensor.hpp>
/**
    @brief ReLU - простая функция активации - выпрямленная линейная функция.
*/
template<typename T>
class ReLU : public Successor<T> {
public:
    ReLU() = delete;
    ReLU(const ReLU&) = delete;
    explicit ReLU(typename ANN<T>::sPtr ann) : Successor<T>(ann), X_{nullptr}, Y_{nullptr}, dX_{nullptr}, dY_{nullptr} {
    };
    ~ReLU() = default;
    void build(typename IBackendFactory<T>::sPtr factory) override {
	Successor<T>::build(factory);
	X_=Successor<T>::getInputs();
	Y_=Successor<T>::getOutputs();
	dX_=Successor<T>::getInputErrors();
	dY_=Successor<T>::getOutputErrors();
    };


    void forward() override {
	assert(X_);
	assert(Y_);
	for(size_t i=0; i<X_->size(); i++)
	    Y_->raw(i) = X_->raw(i) > T(0) ? X_->raw(i) : T(0);
    };
    void backward() override {
	assert(dX_);
	assert(dY_);
	for(size_t i=0; i<X_->size(); i++)
	    dX_->raw(i) =  X_->raw(i) > T(0)? dY_->raw(i) : T(0);
    };
    void batchBegin() override {
	assert(X_);
	assert(Y_);
	assert(dX_);
	assert(dY_);
    };
    void batchEnd() override {
    };
    void setTutor(typename AbstractTutor<T>::uPtr) override {
    };


private:
    TensorPtr<T> X_, Y_;
    TensorPtr<T> dX_, dY_;
protected:
};

#endif