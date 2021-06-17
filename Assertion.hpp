#ifndef __ANN_ASSERTION_HPP__
#define __ANN_ASSERTION_HPP__

#include <stdexcept>
#include <string>
#include <memory>
#include <vector>
#include <functional>

#include <ANN.hpp>
#include <Successor.hpp>
#include <DataHolder.hpp>
#include <AbstractTutor.hpp>
#include <TensorMath.hpp>

/** 
    @brief Assertion - Класс, провереяющий данные, проходящие через него.
    класса Model.
*/
template<typename T>
class Assertion : public Successor<T> {
public:
    Assertion() = delete;
    Assertion(const Assertion&) = delete;
    explicit Assertion(ANN<T>* ann, const std::function<void(T)>& fwd, const std::function<void(T)>& bwd=[](T){} ) : 
	fwd_(fwd), 
	bwd_(bwd), 
	Successor<T>(ann) {
	X_=Successor<T>::getInputs();
	Y_=Successor<T>::getOutputs();
	dX_=Successor<T>::getInputErrors();
	dY_=Successor<T>::getOutputErrors();
    };
    ~Assertion() = default;


    void forward() override {
	for(size_t i=0; i<X_->size(); i++)
	    fwd_(X_->raw(i));
	tensormath::copy<T>(X_, Y_);
    };
    void backward() override {
	for(size_t i=0; i<X_->size(); i++)
	    bwd_(dY_->raw(i));
	tensormath::copy<T>(dY_, dX_);
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
    std::function<void(T)> fwd_, bwd_;
protected:
};

#endif