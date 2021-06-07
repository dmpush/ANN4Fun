#ifndef __MODEL_HPP__
#define __MODEL_HPP__

#include <deque>
#include <vector>
#include <memory>
#include <concepts>
#include <stdexcept>

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


    template<Derived<Succession<T>> AnnType>
    void addLayer() {
	layers_.push_back(std::make_shared<AnnType>(layers_.back().get() ) );
    };

    void forward() override {
	for(auto it: layers_)
	    it->forward();
    };

    void backward() override {
	ANN<T>::backward();
	for(auto it=layers_.rbegin(); it!=layers_.rend(); it++)
	    (*it)->backward();

    };
    void batchBegin() override {
	ANN<T>::batchBegin();
	for(auto it: layers_)
	    it->batchBegin();
    };

    void batchEnd() override {
	for(auto it: layers_)
	    it->batchEnd();
    };

    T getOutput(size_t ind) override { return layers_.back()->getOutput(ind); };
    T getInput (size_t ind) override { return layers_.front()->getInput(ind); };

    T setOutput(size_t ind, T val) override { return layers_.back()->setOutput(ind, val); };
    T setInput (size_t ind, T val) override { return layers_.front()->setInput(ind, val); };

    T setError   (size_t ind, T val) override { return layers_.back()->setError(ind, val); };
    T appendError(size_t ind, T val) override { return layers_.back()->appendError(ind, val); };

    Tensor<T> getInputs()       override { return layers_.front()->getInputs(); };
    Tensor<T> getInputErrors()  override { return layers_.front()->getInputErrors(); };
    Tensor<T> getOutputs()      override { return layers_.back()->getOutputs(); };
    Tensor<T> getOutputErrors() override { return layers_.back()->getOutputErrors(); };

    size_t getNumInputs()  override { return layers_.front()->getNumInputs(); };
    size_t getNumOutputs() override { return layers_.back()->getNumOutputs(); };

    void setMode(typename ANN<T>::WorkModes mode) override {
	ANN<T>::setMode(mode);
	for(auto it: layers_)
	    it->setMode(mode);
    };

    typename ANN<T>::sPtr operator()(int index) {
	return layers_[index];
    };

    void setTutor(typename AbstractTutor<T>::uPtr) override {
	throw std::runtime_error("Model::setupTutor() не доступен");
    };
    

private:
    std::deque<typename ANN<T>::sPtr> layers_;
};

#endif


