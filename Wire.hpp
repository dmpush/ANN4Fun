#ifndef __WIRE_HPP__
#define __WIRE_HPP__

#include <stdexcept>
#include <string>
#include <memory>
#include <vector>

#include <ANN.hpp>
#include <Successor.hpp>
#include <DataHolder.hpp>
#include <AbstractTutor.hpp>
#include <TensorMath.hpp>

/// Простой класс соединительных "проводов".
template<typename T>
class Wire : public Successor<T> {
public:
    Wire() = delete;
    Wire(const Wire&) = delete;
    explicit Wire(ANN<T>* ann) : Successor<T>(ann) {
	X_=Successor<T>::getInputs();
	Y_=Successor<T>::getOutputs();
	dX_=Successor<T>::getInputErrors();
	dY_=Successor<T>::getOutputErrors();
    };
    ~Wire() = default;


    void forward() override {
	tensormath::copy<T>(X_, Y_);
    };
    void backward() override {
	tensormath::copy<T>(dY_, dX_);
	Successor<T>::backward();
    };
    void batchBegin() override {
	ANN<T>::batchBegin();
    };
    void batchEnd() override {
    };
    

    void setTutor(typename AbstractTutor<T>::uPtr tutor) override { 
	throw std::runtime_error("Wire::setTutor() не поддерживатеся");
    };

private:
    Tensor<T> X_, Y_;
    Tensor<T> dX_, dY_;
protected:
};

#endif