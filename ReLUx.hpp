#ifndef __RELU_HPP__
#define __RELU_HPP__

#include <cassert>
#include <stdexcept>
#include <string>
#include <memory>
#include <vector>

#include <ANN.hpp>
#include <Successor.hpp>
#include <DataHolder.hpp>
#include <AbstractTutor.hpp>
#include <Tensor.hpp>
/**
    @brief ReLUx - вариан LeakyReLU с параметрами подогнанными таким образом, чтобы сохранялось нулевое среднее 
    и единичная сигма входного распределения.
*/
template<typename T>
class ReLUx : public Successor<T> {

public:
    ReLUx() = delete;
    ReLUx(const ReLUx&) = delete;
    explicit ReLUx(ANN<T>* ann) : Successor<T>(ann),
	    offset_{0.77689267459685207307},
	    scale_{2.68442669991475435509} {
	X_=Successor<T>::getInputs();
	Y_=Successor<T>::getOutputs();
	dX_=Successor<T>::getInputErrors();
	dY_=Successor<T>::getOutputErrors();
    };
    ~ReLUx() = default;


    void forward() override {
	for(size_t i=0; i<X_->size(); i++) {
	    Y_->raw(i) = X_->raw(i) > offset_ ? (X_->raw(i)-offset_)*scale_ : (X_->raw(i)-offset_)/scale_;
	};
    };
    void backward() override {
	for(size_t i=0; i<X_->size(); i++)
	    dX_->raw(i) =  (X_->raw(i) > offset_ ?  scale_ : T(1.0)/scale_) * dY_->raw(i) ;
    };
    void batchBegin() override {
    };
    void batchEnd() override {
    };
    void setTutor(typename AbstractTutor<T>::uPtr) override {
    };


private:
    TensorPtr<T> X_, Y_;
    TensorPtr<T> dX_, dY_;
    T offset_, scale_;
protected:
};

#endif