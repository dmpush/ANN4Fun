#ifndef __SELU_HPP__
#define __SELU_HPP__

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
    @brief SELU - Scaled Exponential linear unit.
*/
template<typename T>
class SELU : public Successor<T> {
public:
    SELU() = delete;
    SELU(const SELU&) = delete;
    explicit SELU(typename ANN<T>::sPtr ann) : Successor<T>(ann), 
    X_{nullptr},
    Y_{nullptr},
    dX_{nullptr},
    dY_{nullptr},
    alpha_(1.67326), lambda_(1.0507) {
    };
    ~SELU() = default;
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
    inline double f(double x) { return lambda_*(x<0.0 ? alpha_*(std::exp(x)-1.0) : x); };
    inline double df(double x) { return lambda_*(x<0.0 ? alpha_*std::exp(x) : 1.0 ); };
    TensorPtr<T> X_, Y_;
    TensorPtr<T> dX_, dY_;
    double alpha_;
    double lambda_;
protected:
};

#endif