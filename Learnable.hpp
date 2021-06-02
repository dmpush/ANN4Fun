#ifndef __LEARNABLE_HPP__
#define __LEARNABLE_HPP__

#include <stdexcept>
#include <string>
#include <memory>
#include <ANN.hpp>
#include <Successor.hpp>
#include <DataHolder.hpp>
#include <AbstractTutor.hpp>

template<typename T>
class Learnable : public Successor<T> {
public:
    Learnable() = delete;
    Learnable(const Learnable&) = delete;
    explicit Learnable(ANN<T>* ann, size_t Nout) : Successor<T>(ann,{Nout}) {
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
	ANN<T>::batchEnd();
	if(ANN<T>::isTrainable())
	    tutor_->batchEnd();
    };
    virtual void  setupTutor(typename AbstractTutor<T>::uPtr) = 0;

    void setContext(typename DataHolder<T>::sPtr params, typename DataHolder<T>::sPtr grad) {
	tutor_->setContext(params, grad);
    };

private:
    // unique гарантирует невозможность задать одного Учителя нескольким сетям
    typename AbstractTutor<T>::uPtr tutor_;
    // возможно, сюда следует перенести params_, grad
protected:
    void setTutor(typename AbstractTutor<T>::uPtr tutor) { tutor_=std::move(tutor); };
};

#endif