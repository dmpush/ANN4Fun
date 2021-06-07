#ifndef __ANN_HPP__
#define __ANN_HPP__

#include <stdexcept>
#include <memory>
#include <DataHolder.hpp>
#include <AbstractTutor.hpp>
/**
    @brief ANN Суперкласс нейронной сети.
*/
template<typename T>
class ANN {
public:
    using sPtr=std::shared_ptr<ANN<T>>;
    enum WorkModes{TrainMode, WorkMode, UnknownMode};
    ANN() : lockTrain_{false}, mode_{UnknownMode} {};
    ANN(ANN*) {};
    virtual ~ANN() = default;

    virtual void setMode(WorkModes mode) { mode_=mode; };
    WorkModes getMode() { return mode_; };

    void lockTrain() { lockTrain_=true; };
    void unlockTrain() { lockTrain_=false; };
    bool isTrainable() { return !lockTrain_; };


    virtual T getOutput(size_t ind)        final { return getOutputs()->raw(ind); };
    virtual T setOutput(size_t ind, T val) final { return (getOutputErrors()->raw(ind)=val-getOutputs()->raw(ind)); };

    virtual T getInput(size_t ind)         final { return getInputs()->raw(ind); };
    virtual T setInput(size_t ind, T val)  final { return (getInputs()->raw(ind)=val); };

    virtual T setError(size_t ind, T val)  final { return (getOutputErrors()->raw(ind)=val); };
    virtual T appendError(size_t ind, T val)  final { return (getOutputErrors()->raw(ind)+=val); };

    virtual Tensor<T> getInputs()=0;
    virtual Tensor<T> getOutputs()=0;
    virtual Tensor<T> getInputErrors()=0;
    virtual Tensor<T> getOutputErrors()=0;

    virtual size_t getNumInputs()  final { return getInputs()->size(); };
    virtual size_t getNumOutputs() final { return getOutputs()->size(); };
    virtual void forward()=0; ///< прямое распространение сигналов по сети
    /// обратное распространение сигналов по сети
    virtual void backward() { 
	if(mode_==UnknownMode)
	    throw std::runtime_error("ANN::backward(): Не задан режим работы ИНС");
    };
    /// начало батча - обнуление аккумулятора градиента
    virtual void batchBegin() { 
	if(mode_==UnknownMode)
	    throw std::runtime_error("ANN::batchBegin(): Не задан режим работы ИНС");
    };

    virtual void batchEnd() =0; ///< конец батча - здесь происходит обучение
    /// назначение Учителя
    virtual void setTutor(typename AbstractTutor<T>::uPtr) {}; 

private:
    WorkModes mode_;
    bool lockTrain_;
};

#endif