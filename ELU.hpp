#ifndef __ELU_HPP__
#define __ELU_HPP__

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
    @brief ELU - Exponential linear unit.
*/
template<typename T>
class ELU : public Successor<T> {
public:
    ELU() = delete;
    ELU(const ELU&) = delete;
    explicit ELU(ANN<T>* ann, double alpha=1.0) : Successor<T>(ann), alpha_(alpha) {
	X_=Successor<T>::getInputs();
	Y_=Successor<T>::getOutputs();
	dX_=Successor<T>::getInputErrors();
	dY_=Successor<T>::getOutputErrors();
    };
    ~ELU() = default;


    void forward() override {
	for(size_t i=0; i<X_->size(); i++) {
	    double x=X_->raw(i);
	    Y_->raw(i) = x < 0.0 ? alpha_*(std::exp(x) - 1.0) : x;
	};
    };
    void backward() override {
	for(size_t i=0; i<X_->size(); i++) {
	    double x=X_->raw(i);
	    double y=Y_->raw(i);
	    dX_->raw(i) =  ( x < 0 ? alpha_+ y : 1.0 ) * dY_->raw(i);
	};
    };
    void batchBegin() override {
    };
    void batchEnd() override {
    };
    void setTutor(typename AbstractTutor<T>::uPtr) override {
    };


private:
    Tensor<T> X_, Y_;
    Tensor<T> dX_, dY_;
    double alpha_;
protected:
};

#endif