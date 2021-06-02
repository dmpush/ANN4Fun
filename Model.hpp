#ifndef __MODEL_HPP__
#define __MODEL_HPP__

#include <deque>

#include <ANN.hpp>
#include <DataHolder.hpp>

template<typename T>
class Model : public ANN<T> {
public:
    Model() : layers_{} {};
    ~Model() = default;
    // подумать, можно ли тут использовать шаблонную функцию
    // дабы спрятать выделение памяти от пользователя
    void addLayer(ANN<T> * layer) { layers_.push_back(layer); };

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
	ANN<T>::batchEnd();
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


private:
    std::deque<ANN<T>*> layers_;
};

#endif


