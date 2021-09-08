#ifndef __MODEL_HPP__
#define __MODEL_HPP__

#include <deque>
#include <vector>
#include <memory>
#include <concepts>
#include <stdexcept>
#include <iostream>

#include <ANN.hpp>
#include <IBackendFactory.hpp>
#include <IDataHolder.hpp>
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
    using sPtr=std::shared_ptr<Model<T>>;

    Model() =delete; 
//: layers_{} {
//    };
    Model(const std::vector<size_t>& inputShape) {
	auto inputs=std::make_shared<Input<T>>(inputShape);
	layers_.push_back(inputs);
    };


    /// @brief конструктор композиции
    explicit Model(typename ANN<T>::sPtr head) : Succession<T>(head), layers_{} {
	layers_.push_back(std::make_shared<Wire<T>>(head));
    };

    
    ~Model() = default;

    void build(typename IBackendFactory<T>::sPtr factory) override {
	for(auto it: layers_)
	    it->build(factory);
    };

    // дабы спрятать выделение памяти от пользователя
    template<Derived<Succession<T>> AnnType>
    void addLayer(const std::vector<size_t>& dims) {
	layers_.push_back(std::make_shared<AnnType>(layers_.back(), dims) );
    };


    template<Derived<Succession<T>> AnnType, typename... Args>
    void addLayer(Args... args) {
	layers_.push_back(std::make_shared<AnnType>(layers_.back(), args... ) );
    };

    void addLayer(typename ANN<T>::sPtr ann) {
	layers_.push_back(ann);
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


    TensorPtr<T> getInputs()       override { return layers_.front()->getInputs(); };
    TensorPtr<T> getInputErrors()  override { return layers_.front()->getInputErrors(); };
    TensorPtr<T> getOutputs()      override { return layers_.back()->getOutputs(); };
    TensorPtr<T> getOutputErrors() override { return layers_.back()->getOutputErrors(); };

    typename ANN<T>::sPtr operator()(int index) {
	return layers_[index];
    };


    template<Derived<AbstractTutor<T>> Tut, typename... Args>
    void setTutor(Args... args) {
	for(auto it: layers_) {
	    it->setTutor(std::make_unique<Tut>(args...));
	};
    };

    void setTutor(typename AbstractTutor<T>::uPtr tutor) override final {
//	throw std::runtime_error("Невозможно установить одного Учителя для всех компонент сети!");
	for(auto it: layers_) {
	    it->setTutor(std::move(tutor->clone()));
	};
    };

    void dump() override {
	std::cout<<"Model:"<<std::endl;
	for(auto it: layers_) {
	    it->dump();
	};
	getInputs()->dump();
	getOutputs()->dump();
    };
    void notify(typename ANN<T>::Notification* notice) override {
        for(auto l: layers_)
            l->notify(notice);
    };

    std::vector<size_t> shape() override { return layers_.back()->shape(); };
private:
    /// список слоев, составляющих модель
    std::deque<typename ANN<T>::sPtr> layers_;
};

#endif


