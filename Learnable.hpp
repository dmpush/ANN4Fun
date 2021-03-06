#ifndef __LEARNABLE_HPP__
#define __LEARNABLE_HPP__

#include <stdexcept>
#include <string>
#include <memory>
#include <vector>

#include <ANN.hpp>
#include <Successor.hpp>
#include <IBackendFactory.hpp>
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
    explicit Learnable(typename ANN<T>::sPtr ann, std::vector<size_t> Nout) :
	Successor<T>(ann,Nout),
	params_{nullptr},
	grad_{nullptr} {
    };
    /// @brief конструктор для параметризированных функций активации
    explicit Learnable(typename ANN<T>::sPtr ann) : Successor<T>(ann),
	params_{nullptr},
	grad_{nullptr} {
    };
    ~Learnable() = default;

    void build(typename IBackendFactory<T>::sPtr factory) override {
	Successor<T>::build(factory);
	params_=factory->makeHolderS();
    };


    auto getTutor() { return tutor_; };

    void backward() override {
	assert(params_);
	assert(grad_ );
	tutor_->backward();
    };

    void batchBegin() override final {
	assert(params_);
	assert(grad_ );
	tutor_->batchBegin();
    };
    void batchEnd() override final {
	assert(params_);
	assert(grad_ );
	if(ANN<T>::isTrainable())
	    tutor_->batchEnd();
    };
    
    typename IDataHolder<T>::sPtr getParams() { return params_; };
    typename IDataHolder<T>::sPtr getGrad()   { return grad_; };


    void setTutor(typename AbstractTutor<T>::uPtr tutor) override { 
	assert(params_);
	tutor_=std::move(tutor); 
        grad_=params_->clone();
	grad_->fill(T(0));
	tutor_->setContext(params_, grad_);
    };


private:
    // unique гарантирует невозможность задать одного Учителя нескольким сетям
    typename AbstractTutor<T>::uPtr tutor_;
    typename IDataHolder<T>::sPtr params_;
    typename IDataHolder<T>::sPtr grad_;
protected:
};

#endif