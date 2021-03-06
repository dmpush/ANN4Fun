#ifndef __ANN_ASSERTION_HPP__
#define __ANN_ASSERTION_HPP__

#include <stdexcept>
#include <string>
#include <memory>
#include <vector>
#include <functional>

#include <ANN.hpp>
#include <Successor.hpp>
#include <AbstractTutor.hpp>
#include <IBackendFactory.hpp>

/** 
    @brief Assertion - Класс, провереяющий данные, проходящие через него.
    класса Model.
*/
template<typename T>
class Assertion : public Successor<T> {
public:
    Assertion() = delete;
    Assertion(const Assertion&) = delete;
    explicit Assertion(typename ANN<T>::sPtr ann, const std::function<void(T)>& fwd, const std::function<void(T)>& bwd=[](T){} ) : 
	Successor<T>(ann),
	X_{nullptr},
	Y_{nullptr},
	dX_{nullptr},
	dY_{nullptr},
	fwd_(fwd), 
	bwd_(bwd) {};
    ~Assertion() = default;
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
	    fwd_(X_->raw(i));
	Y_->copy(X_);
    };
    void backward() override {
	assert(dX_);
	assert(dY_);
	for(size_t i=0; i<X_->size(); i++)
	    bwd_(dY_->raw(i));
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
    std::function<void(T)> fwd_, bwd_;
protected:
};

#endif