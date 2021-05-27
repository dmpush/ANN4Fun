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


    T getOutput(size_t index) { return Y_->get(index); };
    T setOutput(size_t index, T value) { return dY_->set(index, value - Y_->get(index)); };

    T getInput(size_t index) { return X_->get(index); };
    T setInput(size_t index, T value) { return X_->set(index, value); };

    T setError(size_t index, T value) { return dY_->set(index, value); };
    T appendError(size_t index, T value) { return dY_->set(index, dY_->get(index) + value); };

    size_t getNumInputs()  { return numberOfInputs_; };
    size_t getNumOutputs() { return numberOfOutputs_; };
    // перевести на unique !
    virtual void setTutor(typename AbstractTutor<T>::sPtr) =0;

private:
    size_t numberOfInputs_, numberOfOutputs_;
    typename DataHolder<T>::sPtr holder_;
//    typename AbstracTutor<T>::sPtr tutor_;
    // возможно, сюда следует перенести params_, grad_, tutor_
    
protected:
    typename DataHolder<T>::Tensor::sPtr X_;
    typename DataHolder<T>::Tensor::sPtr Y_;
    typename DataHolder<T>::Tensor::sPtr dX_;
    typename DataHolder<T>::Tensor::sPtr dY_;
};

#endif