#ifndef __ABSTRACT_TUTOR_HPP__
#define __ABSTRACT_TUTOR_HPP__

#include <DataHolder.hpp>
#include <vector>
#include <memory>
#include <vector>
/**
    @brief AbstractTutor - абстрактный класс Учителя нейронной сети. Учитель оперирует двумя объектами DataHolder - 
    вектором скрытых параметров и градиентом ошибки.
*/
template<typename T>
class AbstractTutor {
protected:
    std::vector<T> lambdas_; ///< Коэфф-ты регуляризации
    size_t sampleCount_; ///< счетчик семплов внутри батча
    typename DataHolder<T>::sPtr param_;///< хранилище параметров сети
    typename DataHolder<T>::sPtr grad_; ///< аккумулятор градиента
public:
    using uPtr=std::unique_ptr<AbstractTutor<T>>;
    AbstractTutor() : param_{}, grad_{}, sampleCount_{0} {};
    AbstractTutor(const std::vector<T>& lambdas) : lambdas_{lambdas} {
    };
    AbstractTutor(const AbstractTutor&) = delete;
    virtual ~AbstractTutor() = default;

    void setContext(typename DataHolder<T>::sPtr param, typename DataHolder<T>::sPtr grad) {
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
    /// @brief Вычисление многочлена регуляризации.
    /// @param x  - параметр param_[i].
    /// @return значение функции регуляризации для параметра x.
    T getRegularization(T x) {
	auto sx= x > T(0.0f) ? T(1.0f) : T(-1.0f);
	auto xp=x;
	auto sum=T(0);
	for(size_t i=0; i<lambdas_.size(); i++) 
	    if(lambdas_[i] > T(0.0f)){
		sum += (i>0 ? xp: sx) * lambdas_[i];
		xp*=x;
	    };
	return sum;
    };
};

#endif