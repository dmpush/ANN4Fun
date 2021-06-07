#ifndef __LEARNABLE_HPP__
#define __LEARNABLE_HPP__

#include <stdexcept>
#include <string>
#include <memory>
#include <vector>

#include <ANN.hpp>
#include <Successor.hpp>
#include <DataHolder.hpp>
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
	params_{std::make_shared<DataHolder<T>>()},
	grad_{std::make_shared<DataHolder<T>>()},
	Successor<T>(ann,Nout) {
    };
    ~Learnable() = default;


    auto getTutor() { return tutor_; };

    void backward() override {
	ANN<T>::backward();
	tutor_->backward();
    };

    void batchBegin() override {
	ANN<T>::batchBegin();
	tutor_->batchBegin();
    };
    void batchEnd() override {
	if(ANN<T>::isTrainable())
	    tutor_->batchEnd();
    };
    
    typename DataHolder<T>::sPtr getParams() { return params_; };
    typename DataHolder<T>::sPtr getGrad()   { return grad_; };


    void setTutor(typename AbstractTutor<T>::uPtr tutor) override final { 
	tutor_=std::move(tutor); 
	grad_->fill(T(0));
	tutor_->setContext(params_, grad_);
    };

private:
    // unique гарантирует невозможность задать одного Учителя нескольким сетям
    typename AbstractTutor<T>::uPtr tutor_;
    const typename DataHolder<T>::sPtr params_;
    const typename DataHolder<T>::sPtr grad_;
protected:
};

#endif