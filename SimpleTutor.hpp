#ifndef __SIMPLE_TUTOR_HPP__
#define __SIMPLE_TUTOR_HPP__

#include <AbstractTutor.hpp>
#include <DataHolder.hpp>
//#include <iostream>
template<typename T>
class SimpleTutor : public AbstractTutor<T> {
    T dt_;
public:
    SimpleTutor(const SimpleTutor&) = delete;
    SimpleTutor(T dt=static_cast<T>(0.1f)) : dt_(dt),  AbstractTutor<T>() {};
    ~SimpleTutor() = default;

    void batchEnd() override {
//	std::cout<<"batch_size="<<AbstractTutor<T>::sampleCount_<<std::endl;
	AbstractTutor<T>::param_ -> update(AbstractTutor<T>::grad_, dt_/static_cast<T>(AbstractTutor<T>::sampleCount_) );
    };
    // static std::uniqie_ptr<> build(T dt=static_cast<T>(0.1f)) ...
};

#endif