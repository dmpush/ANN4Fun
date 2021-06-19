#ifndef __SELU_HPP__
#define __SELU_HPP__

#include <stdexcept>
#include <string>
#include <memory>
#include <vector>
#include <cmath>

#include <ANN.hpp>
#include <Successor.hpp>
#include <DataHolder.hpp>
#include <AbstractTutor.hpp>
#include <TensorMath.hpp>
/**
    @brief SELU - Scaled Exponential linear unit.
*/
template<typename T>
class SELU : public Successor<T> {
public:
    SELU() = delete;
    SELU(const SELU&) = delete;
    explicit SELU(ANN<T>* ann) : Successor<T>(ann), 
    alpha_(1.67326), lambda_(1.0507) {
	X_=Successor<T>::getInputs();
	Y_=Successor<T>::getOutputs();
	dX_=Successor<T>::getInputErrors();
	dY_=Successor<T>::getOutputErrors();
    };
    ~SELU() = default;


    void forward() override {
	for(size_t i=0; i<X_->size(); i++) {
	    Y_->raw(i) = f( X_->raw(i) );
	};
    };
    void backward() override {
	for(size_t i=0; i<X_->size(); i++) {
	    dX_->raw(i) =  df(X_->raw(i)) * dY_->raw(i);
	};
    };
    void batchBegin() override {
    };
    void batchEnd() override {
    };
    void setTutor(typename AbstractTutor<T>::uPtr) override {
    };


private:
    inline double f(double x) { return lambda_*(x<0.0 ? alpha_*(std::exp(x)-1.0) : x); };
    inline double df(double x) { return lambda_*(x<0.0 ? alpha_*std::exp(x) : 1.0 ); };
    Tensor<T> X_, Y_;
    Tensor<T> dX_, dY_;
    double alpha_;
    double lambda_;
protected:
};

#endif