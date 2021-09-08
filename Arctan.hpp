#ifndef __ARCTAN_HPP__
#define __ARCTAN_HPP__

#include <stdexcept>
#include <string>
#include <memory>
#include <vector>
#include <cmath>
#include <numbers>

#include <ANN.hpp>
#include <Successor.hpp>
#include <AbstractTutor.hpp>
#include <IBackendFactory.hpp>

/** 
    @brief Arctan - активаторная функция - арктангенс.f(x)=a*arctan(x/a)
*/
template<typename T>
class Arctan : public Successor<T> {
public:
    Arctan() = delete;
    Arctan(const Arctan&) = delete;
    explicit Arctan(typename ANN<T>::sPtr ann) : Successor<T>(ann), scale_{2.0/std::numbers::pi}  {
    };
    ~Arctan() = default;
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
	for(size_t i=0; i<X_->size(); i++)
	    Y_->raw(i) = std::atan(X_->raw(i)/scale_) * scale_;
    };
    void backward() override {
	assert(dX_);
	assert(dY_);
	for(size_t i=0; i<X_->size(); i++) {
	    auto x = X_->raw(i) / scale_;
	    dX_->raw(i) = dY_->raw(i) / (1.0 + x*x);
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
    /// @brief коэфф-т масштабирования на диапазон (-1,1). Вблизи нуля f(x)~x
    const T scale_;
protected:
};

#endif