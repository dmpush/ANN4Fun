#ifndef __GENERATOR_HPP__
#define __GENERATOR_HPP__

#include <vector>
#include <stdexcept>
#include <random>

#include <ANN.hpp>
#include <DataHolder.hpp>
#include <AbstractTutor.hpp>
/**
    @brief Generator - Входной слой нейронной сети, генерирующий случайный гауссов шум, предназначен для GAN.
*/
template<typename T>
class Generator: public ANN<T> {
public:
    Generator(const std::vector<size_t>& Nin) : random_{}, ANN<T>() {
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
    Generator(ANN<T>*) = delete;
    virtual ~Generator() = default;




    Tensor<T>  getInputs()  override  { return fake_; };
    Tensor<T>  getOutputs() override  { return X_; };
    Tensor<T>  getInputErrors()  override { return fake_; };
    Tensor<T>  getOutputErrors() override { return dX_; };

    void forward() override {
	std::normal_distribution<double> generator{0.0, 1.0};
	for(size_t i=0; i<X_->size(); i++) {
	    X_->raw(i)= static_cast<T>(generator(random_));
	};
    };
    void backward() override {
    };
    void batchBegin() override {
    };
    void batchEnd() override {
    };
    void setTutor(typename AbstractTutor<T>::uPtr) override final {
    };
private:
    // хранилище данных и псевдонимы для тензоров
    typename DataHolder<T>::uPtr holder_;
    Tensor<T> X_;
    Tensor<T> dX_;
    Tensor<T> fake_;
    std::random_device random_;
};

#endif