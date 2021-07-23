#ifndef __SOFTMAX_HPP__
#define __SOFTMAX_HPP__

#include <stdexcept>
#include <string>
#include <memory>
#include <vector>
#include <cmath>

#include <ANN.hpp>
#include <Successor.hpp>
#include <IBackendFactory.hpp>
#include <AbstractTutor.hpp>
/**
    @brief SoftMax - обобщенная логистическая функция.
*/
template<typename T>
class SoftMax : public Successor<T> {
public:
    SoftMax() = delete;
    SoftMax(const SoftMax&) = delete;
    explicit SoftMax(ANN<T>* ann) : 
	Successor<T>(ann), 
	X_{nullptr},
	Y_{nullptr},
	dX_{nullptr},
	dY_{nullptr} {};
    ~SoftMax() = default;

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
	T norm{0};
	for(size_t i=0; i<X_->size(); i++) {
	    Y_->raw(i) = static_cast<T>( std::exp(X_->raw(i)) );
	    assert(!std::isnan(Y_->raw(i)));
	    norm += Y_->raw(i);
	    assert(!std::isnan(norm));
	};
	for(size_t i=0; i<X_->size(); i++) {
	    Y_->raw(i) = Y_->raw(i)/norm;
	}
    };
    void backward() override {
	assert(dX_);
	assert(dY_);
	for(size_t i=0; i<Y_->size(); i++) {
	    T sum{0};
	    for(size_t o=0; o<Y_->size(); o++) {
		T Dio=(i==o ? T(1) : T(0));
		sum += Y_->raw(o) *(Dio - Y_->raw(i)) * dY_->raw(o);
	    };
	    dX_->raw(i) =  sum;
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
protected:
};

#endif