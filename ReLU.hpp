#ifndef __RELU_HPP__
#define __RELU_HPP__

#include <stdexcept>
#include <string>
#include <memory>
#include <vector>

#include <ANN.hpp>
#include <Successor.hpp>
#include <IDataHolder.hpp>
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
    explicit ReLU(ANN<T>* ann) : Successor<T>(ann) {
	X_=Successor<T>::getInputs();
	Y_=Successor<T>::getOutputs();
	dX_=Successor<T>::getInputErrors();
	dY_=Successor<T>::getOutputErrors();
    };
    ~ReLU() = default;


    void forward() override {
	for(size_t i=0; i<X_->size(); i++)
	    Y_->raw(i) = X_->raw(i) > T(0) ? X_->raw(i) : T(0);
    };
    void backward() override {
	for(size_t i=0; i<X_->size(); i++)
	    dX_->raw(i) =  X_->raw(i) > T(0)? dY_->raw(i) : T(0);
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
protected:
};

#endif