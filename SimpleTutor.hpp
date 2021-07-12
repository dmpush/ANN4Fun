#ifndef __SIMPLE_TUTOR_HPP__
#define __SIMPLE_TUTOR_HPP__

#include <vector>
#include <AbstractTutor.hpp>
#include <DataHolder.hpp>
/**
    @brief SimpleTutor - простейшая реализация Учителя - градиентный метод с постоянным шагом.
*/
template<typename T>
class SimpleTutor : public AbstractTutor<T> {
    T dt_;
public:
    SimpleTutor(const SimpleTutor&) = delete;
    SimpleTutor(T dt=static_cast<T>(0.1f)) : AbstractTutor<T>(), dt_(dt)   {};
    SimpleTutor(T dt, const std::vector<T>& lambdas) : AbstractTutor<T>(lambdas), dt_(dt) {};
    ~SimpleTutor() = default;

    void batchEnd() override {
	if(AbstractTutor<T>::getSampleCount() ==0)
	    throw std::runtime_error("Пустой батч. Возможно, не хватает вызова Learnable::backward()");
	T bs=static_cast<T>(AbstractTutor<T>::getSampleCount() );
	size_t len=AbstractTutor<T>::param_ ->size();
	for(size_t i=0; i<len; i++) {
	    T Pi=AbstractTutor<T>::param_ ->raw(i);
	    T Gi=AbstractTutor<T>::grad_->raw(i) / bs;
	    AbstractTutor<T>::param_ ->raw(i) =  Pi + dt_ * (Gi -  AbstractTutor<T>::getRegularization(Pi));
	}
    };
};

#endif