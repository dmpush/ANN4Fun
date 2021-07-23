#ifndef __WIRE_HPP__
#define __WIRE_HPP__

#include <stdexcept>
#include <string>
#include <memory>
#include <vector>

#include <ANN.hpp>
#include <Successor.hpp>
#include <AbstractTutor.hpp>
#include <IBackendFactory.hpp>
#include <ITensor.hpp>
#include <IDataHolder.hpp>

/** 
    @brief Wire - Простой класс соединительных "проводов". Основное назначение - участие в конструкторе композиции
    класса Model.
*/
template<typename T>
class Wire : public Successor<T> {
public:
    Wire() = delete;
    Wire(const Wire&) = delete;
    explicit Wire(ANN<T>* ann) : Successor<T>(ann), X_{nullptr}, Y_{nullptr}, dX_{nullptr}, dY_{nullptr} {
    };
    ~Wire() = default;

    void build(typename IBackendFactory<T>::sPtr factory) override final {
	Successor<T>::getPrecursor()->build(factory);
	Successor<T>::build(factory);
	X_=Successor<T>::getInputs();
	Y_=Successor<T>::getOutputs();
	dX_=Successor<T>::getInputErrors();
	dY_=Successor<T>::getOutputErrors();
    };


    void forward() override {
	assert(Y_);
	Y_->copy(X_);
    };
    void backward() override {
	assert(dX_);
	dX_->copy(dY_);
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