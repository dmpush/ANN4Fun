#ifndef __GENERATOR_HPP__
#define __GENERATOR_HPP__

#include <vector>
#include <stdexcept>

#include <ANN.hpp>
#include <IBackendFactory.hpp>
#include <AbstractTutor.hpp>
#include <ITensor.hpp>
/**
    @brief Generator - Входной слой нейронной сети, генерирующий случайный гауссов шум, предназначен для GAN.
*/
template<typename T>
class Generator: public ANN<T> {
public:
    explicit Generator(const std::vector<size_t>& Nin) : 
	ANN<T>(), 
	shape_{Nin},
	holder_{nullptr},
	X_{nullptr},
	dX_{nullptr},
	fake_{nullptr} { };

    explicit Generator(size_t Nin) : Generator<T>(std::vector({Nin})) {};
    Generator(ANN<T>*) = delete;
    virtual ~Generator() = default;

    void build(typename IBackendFactory<T>::sPtr factory) override {
	// сеть является владельцем своих входов и выходов
	holder_=factory->makeHolderU();
	holder_->append("X", shape_);
	holder_->append("dX", shape_);
	holder_->append("fake");
	holder_->build();
	fake_=holder_->get("fake");
	X_=holder_->get("X");
	dX_=holder_->get("dX");
	holder_->fill(T(0));
    };




    TensorPtr<T>  getInputs()  override  { return fake_; };
    TensorPtr<T>  getOutputs() override  { return X_; };
    TensorPtr<T>  getInputErrors()  override { return fake_; };
    TensorPtr<T>  getOutputErrors() override { return dX_; };

    void forward() override {
	assert(X_);
	X_->gaussianNoise(0.0, 1.0);
    };
    void backward() override {
	assert(dX_);
    };
    void batchBegin() override {
	assert(X_);
	assert(dX_);
	assert(fake_);
    };
    void batchEnd() override {
    };
    void setTutor(typename AbstractTutor<T>::uPtr) override final {
    };
    void dump() override  {
	std::cout<<"Generator:"<<std::endl;
	holder_->dump();
    };
    std::vector<size_t> shape() override { return shape_; }; 
private:
    // хранилище данных и псевдонимы для тензоров
    std::vector<size_t> shape_;
    typename IDataHolder<T>::uPtr holder_;
    TensorPtr<T> X_;
    TensorPtr<T> dX_;
    TensorPtr<T> fake_;
};

#endif