#ifndef __CROSS_ENTROPY_HPP__
#define __CROSS_ENTROPY_HPP__

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
    @brief CrossEntropy - слой кросс-энтропии.
    класса Model.
*/
template<typename T>
class CrossEntropy : public Successor<T> {
public:
    using sPtr=std::shared_ptr<Wire<T>>;
    CrossEntropy() = delete;
    CrossEntropy(const CrossEntropy&) = delete;
    explicit CrossEntropy(typename ANN<T>::sPtr ann) : Successor<T>(ann), X_{nullptr}, Y_{nullptr}, dX_{nullptr}, dY_{nullptr} {
    };
    ~CrossEntropy() = default;

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
        for(size_t i=0; i<dY_->size(); i++)
	dX_->raw(i) = dY_ ->raw(i) / X_->raw(i);
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