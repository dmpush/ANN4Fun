#ifndef __MODEL_HPP__
#define __MODEL_HPP__

#include <deque>
#include <vector>
#include <memory>
#include <concepts>
#include <stdexcept>
#include <iostream>

#include <ANN.hpp>
#include <DataHolder.hpp>
#include <Input.hpp>
#include <Wire.hpp>
#include <Succession.hpp>
#include <AbstractTutor.hpp>

/// концепт наследования, C++20
template<typename D, typename B>
concept Derived= std::is_base_of<B, D>::value;

/**
    @brief Model - нейронная сеть составленная из нескольких слоев сетей разных архитектур.
*/
template<typename T>
class Model : public Succession<T> {
public:

    template<typename R>

    Model() : layers_{} {};
    Model(const std::vector<size_t>& shape) : layers_{}, Succession<T>() {
	layers_.push_back(std::make_shared<Input<T>>(shape));
    };
    /// @brief конструктор композиции
    Model(ANN<T> *ann) : layers_{}, Succession<T>(ann) {
	layers_.push_back(std::make_shared<Wire<T>>(ann));
    };

    
    ~Model() = default;

    // дабы спрятать выделение памяти от пользователя
    template<Derived<Succession<T>> AnnType>
    void addLayer(std::vector<size_t> dims) {
	layers_.push_back(std::make_shared<AnnType>(layers_.back().get(), dims) );
    };


    template<Derived<Succession<T>> AnnType, typename... Args>
    void addLayer(Args... args) {
	layers_.push_back(std::make_shared<AnnType>(layers_.back().get(), args... ) );
    };

    void forward() override {
	for(auto it: layers_)
	    it->forward();
    };

    void backward() override {
	for(auto it=layers_.rbegin(); it!=layers_.rend(); it++)
	    (*it)->backward();

    };
    void batchBegin() override {
	for(auto it: layers_)
	    it->batchBegin();
    };

    void batchEnd() override {
	for(auto it: layers_)
	    it->batchEnd();
    };


    Tensor<T> getInputs()       override { return layers_.front()->getInputs(); };
    Tensor<T> getInputErrors()  override { return layers_.front()->getInputErrors(); };
    Tensor<T> getOutputs()      override { return layers_.back()->getOutputs(); };
    Tensor<T> getOutputErrors() override { return layers_.back()->getOutputErrors(); };

    typename ANN<T>::sPtr operator()(int index) {
	return layers_[index];
    };


    template<Derived<AbstractTutor<T>> Tut, typename... Args>
    void setTutor(Args... args) {
	for(auto it: layers_) {
	    it->setTutor(std::make_unique<Tut>(args...));
	};
    };

    void setTutor(typename AbstractTutor<T>::uPtr) override final {
	throw std::runtime_error("Невозможно установить одного Учителя для всех компонент сети!");
    };

    void dump() override {
	std::cout<<"Model:"<<std::endl;
	for(auto it: layers_) {
	    it->dump();
	};
	getInputs()->dump();
	getOutputs()->dump();
    };
    

private:
    /// список слоев, составляющих модель
    std::deque<typename ANN<T>::sPtr> layers_;
};

#endif


