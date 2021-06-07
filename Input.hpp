#ifndef __INPUT_HPP__
#define __INPUT_HPP__

#include <vector>
#include <stdexcept>
#include <ANN.hpp>
#include <DataHolder.hpp>
#include <AbstractTutor.hpp>
/**
    @brief Input - входной слой нейронной сети, предназначен для передачи данны внутрь сети.
*/
template<typename T>
class Input: public ANN<T> {
public:
    Input(const std::vector<size_t>& Nin) : ANN<T>() {
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
    Input(ANN<T>*) = delete;
    virtual ~Input() = default;




    Tensor<T>  getInputs()  override  { return X_; };
    Tensor<T>  getOutputs() override  { return X_; };
    Tensor<T>  getInputErrors()  override { return dX_; };
    Tensor<T>  getOutputErrors() override { return dX_; };

    T getOutput(size_t index)          override { return  X_->raw(index); };
    T setOutput(size_t index, T value) override { return (dX_->raw(index) = value - X_->raw(index)); };

    T getInput(size_t index)          override { return  X_->raw(index); };
    T setInput(size_t index, T value) override { return (X_->raw(index) = value); };

    T setError(size_t index, T value)    override { return (dX_->raw(index) = value); };
    T appendError(size_t index, T value) override { return (dX_->raw(index) = dX_->raw(index) + value); };


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