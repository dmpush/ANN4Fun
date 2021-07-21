#ifndef __WIRE_HPP__
#define __WIRE_HPP__

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
    @brief Wire - Простой класс соединительных "проводов". Основное назначение - участие в конструкторе композиции
    класса Model.
*/
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
	Y_->copy(X_);
    };
    void backward() override {
	dX_->copy(dY_);
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