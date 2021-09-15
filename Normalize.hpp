#ifndef __NORMALIZE_HPP__
#define __NORMALIZE_HPP__

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
    @brief Normalize
*/
template<typename T>
class Normalize : public Successor<T> {
public:
    Normalize() = delete;
    Normalize(const Normalize&) = delete;
    explicit Normalize(typename ANN<T>::sPtr ann) :
	Successor<T>(ann), 
	X_{nullptr},
	Y_{nullptr},
	dX_{nullptr},
	dY_{nullptr} {};
    ~Normalize() = default;

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
	norm_=T(0);
	for(size_t i=0; i<X_->size(); i++) {
	    auto val=X_->raw(i);
	    norm_ += val*val;
	};
	norm_=std::sqrt(norm_);
	for(size_t i=0; i<X_->size(); i++) {
	    Y_->raw(i) = X_->raw(i)/norm_;
	}
    };
    void backward() override {
	assert(dX_);
	assert(dY_);
	for(size_t i=0; i<Y_->size(); i++) {
	    auto x=X_->raw(i);
	    dX_->raw(i) =  (1.0/norm_ - x*x/(norm_*norm_*norm_)) * dY_->raw(i);
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
    T norm_;
protected:
};

#endif