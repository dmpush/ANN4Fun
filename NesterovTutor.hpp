
#ifndef __NESTEROV_TUTOR_HPP__
#define __NESTEROV_TUTOR_HPP__

#include <vector>
#include <AbstractTutor.hpp>
#include <IDataHolder.hpp>
/**
    @brief NesterovTutor - оптимизация методом Нестерова
*/
template<typename T>
class NesterovTutor : public AbstractTutor<T> {
    T dt_, beta_;
    typename IDataHolder<T>::sPtr velocity_;
public:
    NesterovTutor(const NesterovTutor&) = delete;
    NesterovTutor(T dt, T beta) : 
	AbstractTutor<T>(), 
	dt_(dt), 
	beta_(beta), 
	velocity_{nullptr}   {};
    NesterovTutor(T dt, T beta, const std::vector<T>& lambdas) : 
	AbstractTutor<T>(lambdas), 
	dt_(dt), 
	beta_(beta), 
	velocity_{nullptr} {};
    ~NesterovTutor() = default;

    void setContext(typename IDataHolder<T>::sPtr param, typename IDataHolder<T>::sPtr grad) override {
	AbstractTutor<T>::setContext(param, grad);
	velocity_=grad->clone();
	velocity_->fill();
    };

    void batchEnd() override {
	if(AbstractTutor<T>::getSampleCount() ==0)
	    throw std::runtime_error("Пустой батч. Возможно, не хватает вызова Learnable::backward()");
	T bs=static_cast<T>(AbstractTutor<T>::getSampleCount() );
	this->param_->get("*")->optNesterov(this->grad_->get("*"), velocity_->get("*"), 1.0/bs, dt_, beta_, this->lambdas_);
    };
    typename AbstractTutor<T>::uPtr clone() override {
	auto out=std::make_unique<NesterovTutor<T>>(dt_, beta_, AbstractTutor<T>::lambdas_);
	return out;
    };
};

#endif    
