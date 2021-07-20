#ifndef __LEARNABLE_HPP__
#define __LEARNABLE_HPP__

#include <stdexcept>
#include <string>
#include <memory>
#include <vector>

#include <ANN.hpp>
#include <Successor.hpp>
#include <IDataHolder.hpp>
#include <AbstractTutor.hpp>
/**
    @brief Learnable - абстрактный слой/сеть, содержащий в своем составе Учителя, вектор параметров и градиента.
*/
template<typename T>
class Learnable : public Successor<T> {
public:
    Learnable() = delete;
    Learnable(const Learnable&) = delete;
    explicit Learnable(ANN<T>* ann, std::vector<size_t> Nout) :
	Successor<T>(ann,Nout),
	params_{std::make_shared<DataHolder<T>>()},
	grad_{nullptr} {
    };
    /// @brief конструктор для параметризированных функций активации
    explicit Learnable(ANN<T>* ann) : Successor<T>(ann),
	params_{std::make_shared<DataHolder<T>>()},
	grad_{} {
    };
    ~Learnable() = default;


    auto getTutor() { return tutor_; };

    void backward() override {
	assert( grad_ );
	tutor_->backward();
    };

    void batchBegin() override final {
	tutor_->batchBegin();
    };
    void batchEnd() override final {
	if(ANN<T>::isTrainable())
	    tutor_->batchEnd();
    };
    
    typename IDataHolder<T>::sPtr getParams() { return params_; };
    typename IDataHolder<T>::sPtr getGrad()   { return grad_; };


    void setTutor(typename AbstractTutor<T>::uPtr tutor) override { 
	tutor_=std::move(tutor); 
        grad_=params_->clone();
	grad_->fill(T(0));
	tutor_->setContext(params_, grad_);
    };


private:
    // unique гарантирует невозможность задать одного Учителя нескольким сетям
    typename AbstractTutor<T>::uPtr tutor_;
    const typename IDataHolder<T>::sPtr params_;
    typename IDataHolder<T>::sPtr grad_;
protected:
};

#endif