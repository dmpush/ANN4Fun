#ifndef __ADAM_TUTOR_HPP__
#define __ADAM_TUTOR_HPP__

#include <vector>
#include <AbstractTutor.hpp>
#include <IDataHolder.hpp>
/**
    @brief AdamTutor - оптимизация методом ADAM
*/
template<typename T>
class AdamTutor : public AbstractTutor<T> {
    size_t batchCount_;
    T dt_, beta1_, beta2_, eps_;
    typename IDataHolder<T>::sPtr Mt_, Vt_;
public:
    AdamTutor(const AdamTutor&) = delete;
    AdamTutor(T dt=T(1e-3), T beta1=T(0.9), T beta2=T(0.999), T eps=T(1e-8)) : 
	AbstractTutor<T>(), 
	batchCount_{0},
	dt_{dt},
	beta1_{beta1}, 
	beta2_{beta2}, 
	eps_{eps},
	Mt_{nullptr},
	Vt_{nullptr}   {};
    AdamTutor(T dt, T beta1, T beta2, T eps, const std::vector<T>& lambdas) : 
	AbstractTutor<T>(lambdas),
	batchCount_{0},
	dt_{dt},
	beta1_{beta1}, 
	beta2_{beta2}, 
	eps_{eps},
	Mt_{nullptr},
	Vt_{nullptr} {};
    ~AdamTutor() = default;

    void setContext(typename IDataHolder<T>::sPtr param, typename IDataHolder<T>::sPtr grad) override {
	AbstractTutor<T>::setContext(param, grad);
	Mt_=grad->clone();
	Mt_->fill();
	Vt_=grad->clone();
	Vt_->fill();
    };

    void batchEnd() override {
	if(AbstractTutor<T>::getSampleCount() ==0)
	    throw std::runtime_error("Пустой батч. Возможно, не хватает вызова Learnable::backward()");
	T bs=static_cast<T>(AbstractTutor<T>::getSampleCount() );
	this->param_->get("*")->optAdam(this->grad_->get("*"), Mt_->get("*"), Vt_->get("*"),
	    1.0/bs, static_cast<T>(batchCount_), dt_, beta1_, beta2_, eps_, this->lambdas_);
	batchCount_++;
    };
    typename AbstractTutor<T>::uPtr clone() override {
	auto out=std::make_unique<AdamTutor<T>>(dt_, beta1_, beta2_, eps_, this->lambdas_);
	return out;
    };
};

#endif    
