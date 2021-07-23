#ifndef __INPUT_HPP__
#define __INPUT_HPP__

#include <iostream>
#include <vector>
#include <stdexcept>
#include <ANN.hpp>
#include <IBackendFactory.hpp>
#include <AbstractTutor.hpp>
/**
    @brief Input - Входной слой нейронной сети, предназначен для передачи данных внутрь сети.
*/
template<typename T>
class Input: public ANN<T> {
public:
    Input(const std::vector<size_t>& Nin) : ANN<T>(), input_shape_{Nin}, holder_{nullptr}, X_{nullptr}, dX_{nullptr} {
    };
    Input(ANN<T>*) = delete;
    virtual ~Input() = default;

    void build(typename IBackendFactory<T>::sPtr factory) override {
	// сеть является владельцем своих входов и выходов
	holder_=std::move(factory->makeHolderU());
	holder_->append("X", input_shape_);
	holder_->append("dX", input_shape_);
	holder_->build();
	X_=holder_->get("X");
	dX_=holder_->get("dX");
	holder_->fill(T(0));
    };


    TensorPtr<T>  getInputs()  override  { return X_; };
    TensorPtr<T>  getOutputs() override  { return X_; };
    TensorPtr<T>  getInputErrors()  override { return dX_; };
    TensorPtr<T>  getOutputErrors() override { return dX_; };

    void forward() override {
	assert(X_);
    };
    void backward() override {
	assert(dX_);
    };
    void batchBegin() override {
	assert(X_);
	assert(dX_);
    };
    void batchEnd() override {
    };
    void setTutor(typename AbstractTutor<T>::uPtr) override final {
    };
    void dump()  override {
	std::cout<<"Input:"<<std::endl;
	holder_->dump();
    };
    std::vector<size_t> shape() override { return input_shape_; };
private:
    std::vector<size_t> input_shape_;
    // хранилище данных и псевдонимы для тензоров
    typename IDataHolder<T>::uPtr holder_;
    TensorPtr<T> X_;
    TensorPtr<T> dX_;
};

#endif