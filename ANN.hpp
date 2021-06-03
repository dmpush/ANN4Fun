#ifndef __ANN_HPP__
#define __ANN_HPP__

#include <stdexcept>
#include <memory>
#include <DataHolder.hpp>
#include <AbstractTutor.hpp>

template<typename T>
class ANN {
public:
    using sPtr=std::shared_ptr<ANN<T>>;
    enum WorkModes{TrainMode, WorkMode, UnknownMode};
    ANN() : time_{0}, lockTrain_{false}, mode_{UnknownMode} {};
    virtual ~ANN() = default;

    virtual void setMode(WorkModes mode) { mode_=mode; };
    WorkModes getMode() { return mode_; };

    void lockTrain() { lockTrain_=true; };
    void unlockTrain() { lockTrain_=false; };
    bool isTrainable() { return !lockTrain_; };


    virtual T getOutput(size_t)=0;
    virtual T setOutput(size_t, T)=0;
    virtual T getInput(size_t)=0;
    virtual T setInput(size_t, T)=0;
    virtual T setError(size_t, T)=0;
    virtual T appendError(size_t, T)=0;
    virtual Tensor<T> getInputs()=0;
    virtual Tensor<T> getOutputs()=0;
    virtual Tensor<T> getInputErrors()=0;
    virtual Tensor<T> getOutputErrors()=0;

    virtual size_t getNumInputs()=0;
    virtual size_t getNumOutputs()=0;
    virtual void forward()=0;
    virtual void backward() {
	if(mode_==UnknownMode)
	    throw std::runtime_error("ANN::backward(): Не задан режим работы ИНС");
    };
    virtual void batchBegin() {
	if(mode_==UnknownMode)
	    throw std::runtime_error("ANN::batchBegin(): Не задан режим работы ИНС");
    };

    virtual void batchEnd() =0;
    
    virtual void setTutor(typename AbstractTutor<T>::uPtr) = 0;

private:
    size_t time_;
    WorkModes mode_;
    bool lockTrain_;
};

#endif