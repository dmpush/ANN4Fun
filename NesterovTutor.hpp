#ifndef __NESTEROV_TUTOR_HPP__
#define __NESTEROV_TUTOR_HPP__

#include <vector>
#include <AbstractTutor.hpp>
#include <DataHolder.hpp>
/**
    @brief NesterovTutor - оптимизация методом Нестерова
*/
template<typename T>
class NesterovTutor : public AbstractTutor<T> {
    T dt_, beta_;
    typename DataHolder<T>::uPtr velocity_;
public:
    NesterovTutor(const NesterovTutor&) = delete;
    NesterovTutor(T dt, T beta) : 
	AbstractTutor<T>(), 
	dt_(dt), 
	beta_(beta), 
	velocity_{std::make_unique<DataHolder<T>>()}   {};
    NesterovTutor(T dt, T beta, const std::vector<T>& lambdas) : 
	AbstractTutor<T>(lambdas), 
	dt_(dt), 
	beta_(beta), 
	velocity_{std::make_unique<DataHolder<T>>()} {};
    ~NesterovTutor() = default;

    void setContext(typename DataHolder<T>::sPtr param, typename DataHolder<T>::sPtr grad) override {
	AbstractTutor<T>::setContext(param, grad);
	velocity_->clone(grad);
	velocity_->fill();
    };

    void batchEnd() override {
	if(AbstractTutor<T>::getSampleCount() ==0)
	    throw std::runtime_error("Пустой батч. Возможно, не хватает вызова Learnable::backward()");
	T bs=static_cast<T>(AbstractTutor<T>::getSampleCount() );
	size_t len=AbstractTutor<T>::param_ ->size();
	for(size_t i=0; i<len; i++) {
	    T Vi=velocity_->raw(i);
	    T Pi=AbstractTutor<T>::param_ ->raw(i);
	    T Gi=AbstractTutor<T>::grad_->raw(i) / bs -  AbstractTutor<T>::getRegularization(Pi);
	    Vi = Vi*beta_ + Gi*(1.0 - beta_);
	    AbstractTutor<T>::param_ ->raw(i) =  Pi + dt_ * Vi;
	    velocity_->raw(i) = Vi;
	}
    };
    typename AbstractTutor<T>::uPtr clone() override {
	auto out=std::make_unique<NesterovTutor<T>>(dt_, beta_, AbstractTutor<T>::lambdas_);
	return out;
    };
};

#endif    
