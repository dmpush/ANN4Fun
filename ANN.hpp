#ifndef __ANN_HPP__
#define __ANN_HPP__

#include <stdexcept>
#include <cassert>
#include <memory>
#include <ITensor.hpp>
#include <AbstractTutor.hpp>
/**
    @brief ANN Суперкласс абстрактной нейронной сети.
*/
template<typename T>
class ANN {
public:
    /// @brief Тип умного указателя на абстрактную сеть.
    using sPtr=std::shared_ptr<ANN<T>>;
    /// @brief Конструктор по умолчанию.
    ANN() : lockTrain_{false} {};
    /// @brief Конструктор "следующего слоя".
    ANN(ANN*) : ANN() {};
    /// @brief Деструктор.
    virtual ~ANN() = default;

    /// @brief Блокировка обучения сети/подсети.
    void lockTrain() { lockTrain_=true; };
    /// @brief Разблокировка обучения сети/подсети.
    void unlockTrain() { lockTrain_=false; };
    /// @return true, если обучение разблокировано.
    bool isTrainable() { return !lockTrain_; };

    /// @brief Возвращает значение выхода.
    /// @param ind -- номер выхода.
    /// @return выход с номером ind нейросети.
    virtual T getOutput(size_t ind)        final { return getOutputs()->raw(ind); };
    /// @brief Установить целевое значение для выхода нейросети.
    /// @param ind -- номер выхода;
    /// @param val -- целевое значение.
    /// @return невязка выхода.
    virtual T setOutput(size_t ind, T val) final { 
	assert(ind<getOutputs()->size());
	return (getOutputErrors()->raw(ind)=val-getOutputs()->raw(ind)); 
    };

    /// @brief Возвращает значение входа.
    /// @param ind -- номер входа.
    /// @return значение входа с номером ind нейросети.
    virtual T getInput(size_t ind)         final { return getInputs()->raw(ind); };
    /// @brief Установить вход нейросети.
    /// @param ind -- номер входа;
    /// @param val -- значение входного сигнала.
    /// @return входной сигнал.
    virtual T setInput(size_t ind, T val)  final { 
	assert(ind<getInputs()->size());
	return (getInputs()->raw(ind)=val); 
    };
    /// @brief Установка невязки для заданного выхода нейросети;
    /// @param ind -- номер выхода сети;
    /// @param val -- невязка выхода.
    /// @return невязка выхода.
    virtual T setError(size_t ind, T val)  final { return (getOutputErrors()->raw(ind)=val); };
    /// @brief Добавление к невязке по выходу ind сети значения val.
    /// @param ind -- номер выхода;
    /// @param val -- прибавляемое значение невязки.
    /// @return новое значение невязки по данному выходу.
    virtual T appendError(size_t ind, T val)  final { return (getOutputErrors()->raw(ind)+=val); };
    /// @return входы нейросети в виде тензора.
    virtual TensorPtr<T> getInputs()=0;
    /// @return выходной тензор нейросети.
    virtual TensorPtr<T> getOutputs()=0;
    /// @return тензор входных невязок нейросети.
    virtual TensorPtr<T> getInputErrors()=0;
    /// @return тензор выходных невязок нейросети.
    virtual TensorPtr<T> getOutputErrors()=0;
    /// @return Полное число входов сети.
    virtual size_t getNumInputs()  final { return getInputs()->size(); };
    /// @return Полное число выходов сети.
    virtual size_t getNumOutputs() final { return getOutputs()->size(); };
    /// @brief Прямое распространение сигналов по сети.
    virtual void forward()=0; 
    /// @brief Обратное распространение сигналов по сети.
    virtual void backward()=0;
    /// @brief Начало батча - обнуление аккумулятора градиента.
    virtual void batchBegin()=0;
    /// @brief Конец батча - здесь происходит обучение.
    virtual void batchEnd() =0; 
    /// Назначение Учителя.
    virtual void setTutor(typename AbstractTutor<T>::uPtr) = 0; 
    /// отладочная информация
    virtual void dump() = 0;
    class Notification {
    public:
	virtual ~Notification() = default;
    };
    virtual void notify(Notification*) {};
private:
    /// @brief Приватный флаг блокировки обучения сети.
    bool lockTrain_;
};

#endif