#ifndef __ABSTRACT_TUTOR_HPP__
#define __ABSTRACT_TUTOR_HPP__

#include <DataHolder.hpp>
#include <vector>
#include <memory>

template<typename T>
class AbstractTutor {
protected:
//    std::vector<T> lambdas_;
    size_t sampleCount_;
    typename DataHolder<T>::sPtr param_;
    typename DataHolder<T>::sPtr grad_;
public:
    using sPtr=std::shared_ptr<AbstractTutor<T>>;
    AbstractTutor() = default;
    AbstractTutor(const AbstractTutor&) = delete;
    virtual ~AbstractTutor() = default;

    void setContext(typename DataHolder<T>::sPtr param, typename DataHolder<T>::sPtr grad) {
	sampleCount_ = 0;
	param_ = param;
	grad_ = grad;
	grad_->fill(T(0));
    };
    virtual void batchBegin() {
//	if(grad_.isempty())
//	    throw std::runtime_error("Градиент пуст!");
        sampleCount_=0;
        grad_->fill(T(0));
    };
    virtual void batchEnd() = 0;
    void backward() { sampleCount_++; };
    auto getParams() { return param_; };
    auto getGrad() { return grad_; };
};

#endif