#ifndef __GENERATOR_HPP__
#define __GENERATOR_HPP__

#include <vector>
#include <stdexcept>

#include <ANN.hpp>
#include <DataHolder.hpp>
#include <AbstractTutor.hpp>
#include <ITensor.hpp>
/**
    @brief Generator - Входной слой нейронной сети, генерирующий случайный гауссов шум, предназначен для GAN.
*/
template<typename T>
class Generator: public ANN<T> {
public:
    explicit Generator(const std::vector<size_t>& Nin) : ANN<T>() {
	// сеть является владельцем своих входов и выходов
	holder_=std::make_unique<DataHolder<T>>();
	holder_->append("X", Nin);
	holder_->append("dX", Nin);
	holder_->append("fake");
	holder_->build();
	fake_=holder_->get("fake");
	X_=holder_->get("X");
	dX_=holder_->get("dX");
	holder_->fill(T(0));
//	holder_->description();
    };

    explicit Generator(size_t Nin) : Generator<T>(std::vector({Nin})) {};
    Generator(ANN<T>*) = delete;
    virtual ~Generator() = default;




    TensorPtr<T>  getInputs()  override  { return fake_; };
    TensorPtr<T>  getOutputs() override  { return X_; };
    TensorPtr<T>  getInputErrors()  override { return fake_; };
    TensorPtr<T>  getOutputErrors() override { return dX_; };

    void forward() override {
	X_->gaussianNoise(0.0, 1.0);
    };
    void backward() override {
    };
    void batchBegin() override {
    };
    void batchEnd() override {
    };
    void setTutor(typename AbstractTutor<T>::uPtr) override final {
    };
    void dump() override  {
	std::cout<<"Generator:"<<std::endl;
	holder_->dump();
    };
private:
    // хранилище данных и псевдонимы для тензоров
    typename IDataHolder<T>::uPtr holder_;
    TensorPtr<T> X_;
    TensorPtr<T> dX_;
    TensorPtr<T> fake_;
};

#endif