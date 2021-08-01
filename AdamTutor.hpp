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
    T dt_, beta1_, beta2_, eps_;
    T Beta1_, Beta2_;
    typename IDataHolder<T>::sPtr Mt_, Vt_;
public:
    AdamTutor(const AdamTutor&) = delete;
    AdamTutor(T dt=T(1e-3), T beta1=T(0.9), T beta2=T(0.999), T eps=T(1e-8)) : 
	AbstractTutor<T>(), 
	dt_{dt},
	beta1_{beta1}, 
	beta2_{beta2}, 
	eps_{eps},
	Beta1_{beta1_},
        Beta2_{beta2_},
	Mt_{nullptr},
	Vt_{nullptr}   {};
    AdamTutor(T dt, T beta1, T beta2, T eps, const std::vector<T>& lambdas) : 
	AbstractTutor<T>(lambdas),
	dt_{dt},
	beta1_{beta1}, 
	beta2_{beta2}, 
	eps_{eps},
	Beta1_{beta1_},
        Beta2_{beta2_},
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
	    1.0/bs, Beta1_, Beta2_, dt_, beta1_, beta2_, eps_, this->lambdas_);
	Beta1_*=beta1_;
        Beta2_*=beta2_;
    };
    typename AbstractTutor<T>::uPtr clone() override {
	auto out=std::make_unique<AdamTutor<T>>(dt_, beta1_, beta2_, eps_, this->lambdas_);
	return out;
    };
};

#endif    
