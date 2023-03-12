#ifndef __SiLU_HPP__
#define __SiLU_HPP__

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
    @brief SiLU - Sigmoid-weighed Linear function
*/
template<typename T=float>
class SiLU : public Successor<T> {
public:
    SiLU() = delete;
    SiLU(const SiLU&) = delete;
    explicit SiLU(typename ANN<T>::sPtr ann) : Successor<T>(ann), 
    X_{nullptr},
    Y_{nullptr},
    dX_{nullptr},
    dY_{nullptr} {
    };
    ~SiLU() = default;
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
	    Y_->raw(i) = f( X_->raw(i) );
	};
    };
    void backward() override {
	assert(dX_);
	assert(dY_);
	for(size_t i=0; i<X_->size(); i++) {
	    dX_->raw(i) =  df(X_->raw(i)) * dY_->raw(i);
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
    inline double f(double x) { return  x / (1 + std::exp(-x)); };
    inline double df(double x) { 
	const T ex=std::exp(-x);
	return (1+ex*(1+x))/ ((1+ex)*(1+ex)); 
    };
    TensorPtr<T> X_, Y_;
    TensorPtr<T> dX_, dY_;
protected:
};

#endif