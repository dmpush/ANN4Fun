#ifndef __ABSTRACT_TUTOR_HPP__
#define __ABSTRACT_TUTOR_HPP__

#include <IDataHolder.hpp>
#include <vector>
#include <memory>
#include <vector>
/**
    @brief AbstractTutor - абстрактный класс Учителя нейронной сети. Учитель оперирует двумя объектами DataHolder - 
    вектором скрытых параметров и градиентом ошибки.
*/
template<typename T>
class AbstractTutor {
    size_t sampleCount_; ///< счетчик семплов внутри батча
protected:
    std::vector<T> lambdas_; ///< Коэфф-ты регуляризации
    typename IDataHolder<T>::sPtr param_;///< хранилище параметров сети
    typename IDataHolder<T>::sPtr grad_; ///< аккумулятор градиента
public:
    using uPtr=std::unique_ptr<AbstractTutor<T>>;
    AbstractTutor() : sampleCount_{0}, param_{}, grad_{}  {};
    AbstractTutor(const std::vector<T>& lambdas) : lambdas_{lambdas} {
    };
    AbstractTutor(const AbstractTutor&) = delete;
    virtual ~AbstractTutor() = default;

    virtual void setContext(typename IDataHolder<T>::sPtr param, typename IDataHolder<T>::sPtr grad) {
	sampleCount_ = 0;
	param_ = param;
	grad_ = grad;
	grad_->fill(T(0));
    };
    /// @brief Функция начала батча -- обнуляет счетчик семплов и аккумулятор градиента.
    virtual void batchBegin() {
	if(grad_->isEmpty())
	    throw std::runtime_error("Градиент пуст!");
        sampleCount_=0;
        grad_->fill(T(0));
    };
    /// @brief Метод Шага обучения -- ее каждый Учитель определяет самостоятельно.
    virtual void batchEnd() = 0;
    /// @brief Функция backward просто ведет подсчет семплов в батче.
    void backward() { sampleCount_++; };
    /// @brief Массив парметров, за который отвечает Учитель.
    /// @return указатель на холдер параметров.
    auto getParams() { return param_; };
    /// @brief Градиент параметров.
    /// @return аккумулятор градиента параметров.
    auto getGrad() { return grad_; };

    /// @brief счетчик семплов в батче.
    /// @returns число семплов в батче
    size_t getSampleCount() const { return sampleCount_; };
    /// @brief поддержка паттерна Прототип для смены Учителя на-лету
    /// @return указатель unique_ptr на новосозданный объект
    virtual uPtr clone()  = 0;
};

#endif
