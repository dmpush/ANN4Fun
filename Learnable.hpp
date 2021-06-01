#ifndef __LEARNABLE_HPP__
#define __LEARNABLE_HPP__

#include <stdexcept>
#include <string>
#include <memory>
#include <ANN.hpp>
#include <DataHolder.hpp>
#include <AbstractTutor.hpp>

template<typename T>
class Learnable : public ANN<T> {
public:
    Learnable() = delete;
    Learnable(const Learnable&) = delete;
    explicit Learnable(size_t Nin, size_t Nout) : 
	numberOfInputs_(Nin), 
	numberOfOutputs_(Nout),
	ANN<T>() {
	holder_=std::make_shared<DataHolder<T>>();
	holder_->append("X", {Nin});
	holder_->append("dX", {Nin});
	holder_->append("Y", {Nout});
	holder_->append("dY", {Nout});
	holder_->build();

	X_=holder_->get("X");
	dX_=holder_->get("dX");
	Y_=holder_->get("Y");
	dY_=holder_->get("dY");
	holder_->fill(T(0));
    };
    ~Learnable() = default;

    Tensor<T>  getInputs()  { return X_; };
    Tensor<T>  getOutputs() { return Y_; };

    T getOutput(size_t index) { return Y_->get(index); };
    T setOutput(size_t index, T value) { return dY_->set(index, value - Y_->get(index)); };

    T getInput(size_t index) { return X_->get(index); };
    T setInput(size_t index, T value) { return X_->set(index, value); };

    T setError(size_t index, T value) { return dY_->set(index, value); };
    T appendError(size_t index, T value) { return dY_->set(index, dY_->get(index) + value); };

    size_t getNumInputs()  { return numberOfInputs_; };
    size_t getNumOutputs() { return numberOfOutputs_; };

    auto getTutor() { return tutor_; };

    void backward() override {
	ANN<T>::backward();
	tutor_->backward();
    };

    void batchBegin() override {
	ANN<T>::batchBegin();
	tutor_->batchBegin();
    };
    void batchEnd() override {
	ANN<T>::batchEnd();
	if(ANN<T>::isTrainable())
	    tutor_->batchEnd();
    };
    virtual void  setupTutor(typename AbstractTutor<T>::uPtr) = 0;

    void setContext(typename DataHolder<T>::sPtr params, typename DataHolder<T>::sPtr grad) {
	tutor_->setContext(params, grad);
    };

private:
    size_t numberOfInputs_, numberOfOutputs_;
    typename DataHolder<T>::sPtr holder_;
    typename AbstractTutor<T>::uPtr tutor_;
    // возможно, сюда следует перенести params_, grad_, tutor_
    
    Tensor<T> X_;
    Tensor<T> Y_;
protected:
    void setTutor(typename AbstractTutor<T>::uPtr tutor) { tutor_=std::move(tutor); };
    Tensor<T> dX_;
    Tensor<T> dY_;
};

#endif