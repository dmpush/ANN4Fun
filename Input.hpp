#ifndef __INPUT_HPP__
#define __INPUT_HPP__

#include <iostream>
#include <vector>
#include <stdexcept>
#include <ANN.hpp>
#include <DataHolder.hpp>
#include <AbstractTutor.hpp>
/**
    @brief Input - Входной слой нейронной сети, предназначен для передачи данных внутрь сети.
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




    TensorPtr<T>  getInputs()  override  { return X_; };
    TensorPtr<T>  getOutputs() override  { return X_; };
    TensorPtr<T>  getInputErrors()  override { return dX_; };
    TensorPtr<T>  getOutputErrors() override { return dX_; };

    void forward() override {
    };
    void backward() override {
    };
    void batchBegin() override {
    };
    void batchEnd() override {
    };
    void setTutor(typename AbstractTutor<T>::uPtr) override final {
    };
    void dump()  override {
	std::cout<<"Input:"<<std::endl;
	holder_->dump();
    };
private:
    // хранилище данных и псевдонимы для тензоров
    typename DataHolder<T>::uPtr holder_;
    TensorPtr<T> X_;
    TensorPtr<T> dX_;
};

#endif