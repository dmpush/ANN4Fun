#ifndef __ELU_HPP__
#define __ELU_HPP__

#include <stdexcept>
#include <string>
#include <memory>
#include <vector>
#include <cmath>

#include <ANN.hpp>
#include <Successor.hpp>
#include <IBackendFactory.hpp>
#include <AbstractTutor.hpp>
#include <ITensor.hpp>
/**
    @brief ELU - Exponential linear unit.
*/
template<typename T>
class ELU : public Successor<T> {
public:
    ELU() = delete;
    ELU(const ELU&) = delete;
    explicit ELU(typename ANN<T>::sPtr ann, double alpha=1.0) : Successor<T>(ann), 
    X_{nullptr},
    Y_{nullptr},
    dX_{nullptr},
    dY_{nullptr},
    alpha_(alpha) {
    };
    ~ELU() = default;
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
	for(size_t i=0; i<X_->size(); i++) {
	    double x=X_->raw(i);
	    Y_->raw(i) = x < 0.0 ? alpha_*(std::exp(x) - 1.0) : x;
	};
    };
    void backward() override {
	assert(dX_);
	assert(dY_);
	for(size_t i=0; i<X_->size(); i++) {
	    double x=X_->raw(i);
	    double y=Y_->raw(i);
	    dX_->raw(i) =  ( x < 0 ? alpha_+ y : 1.0 ) * dY_->raw(i);
	};
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
    double alpha_;
protected:
};

#endif