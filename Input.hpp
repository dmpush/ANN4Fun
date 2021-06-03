#ifndef __INPUT_HPP__
#define __INPUT_HPP__

#include <vector>
#include <stdexcept>
#include <ANN.hpp>
#include <DataHolder.hpp>
#include <AbstractTutor.hpp>

template<typename T>
class Input: public ANN<T> {
public:
    Input(std::vector<size_t> Nin) : ANN<T>() {
	// сеть является владельцем своих входов и выходов
	holder_=std::make_unique<DataHolder<T>>();
	holder_->append("X", Nin);
	holder_->append("dX", Nin);
	holder_->build();
	X_=holder_->get("X");
	dX_=holder_->get("dX");
	holder_->fill(T(0));
//	holder_->description();
    };
    virtual ~Input() = default;




    Tensor<T>  getInputs()  override  { return X_; };
    Tensor<T>  getOutputs() override  { return X_; };
    Tensor<T>  getInputErrors()  override { return dX_; };
    Tensor<T>  getOutputErrors() override { return dX_; };

    T getOutput(size_t index)          override { return X_->get(index); };
    T setOutput(size_t index, T value) override { return dX_->set(index, value - X_->get(index)); };

    T getInput(size_t index)          override { return X_->get(index); };
    T setInput(size_t index, T value) override { return X_->set(index, value); };

    T setError(size_t index, T value)    override { return dX_->set(index, value); };
    T appendError(size_t index, T value) override { return dX_->set(index, dX_->get(index) + value); };


    size_t getNumInputs()  override { return X_->size(); };
    size_t getNumOutputs() override { return X_->size(); };

    void forward() override {
    };
    void batchEnd() override {
    };
    void setTutor(typename AbstractTutor<T>::uPtr) override {
    };
private:
    // хранилище данных и псевдонимы для тензоров
    typename DataHolder<T>::uPtr holder_;
    Tensor<T> X_;
    Tensor<T> dX_;
};

#endif