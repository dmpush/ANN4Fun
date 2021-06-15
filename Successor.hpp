#ifndef __SUCCESSOR_HPP__
#define __SUCCESSOR_HPP__

#include <iostream>
#include <vector>
#include <stdexcept>
#include <ANN.hpp>
#include <DataHolder.hpp>
#include <Succession.hpp>

/**
    @brief Successor - сеть, которая явяется очередным слоем в составе другой сети. Имеет собственную память - 
    выходные значения и сигналы ошибок.
*/
template<typename T>
class Successor : public Succession<T> {
public:
    Successor(ANN<T> *ann, std::vector<size_t> Nout) : precursor_(ann), Succession<T>(ann) {
	// сеть является владельцем своих входов и выходов
	holder_=std::make_unique<DataHolder<T>>();
	holder_->append("Y", Nout);
	holder_->append("dY", Nout);
	holder_->build();
	// псевдонимы
	X_ = precursor_ -> getOutputs();
	dX_= precursor_ -> getOutputErrors();
	Y_ = holder_    -> get("Y");
	dY_= holder_    -> get("dY");
	holder_->fill(T(0));
    };

    Successor(ANN<T> *ann) : Successor(ann, ann->getOutputs()->dims()) {};

    virtual ~Successor() = default;




    Tensor<T>  getInputs()  override  { return X_; };
    Tensor<T>  getOutputs() override  { return Y_; };
    Tensor<T>  getInputErrors()  override { return dX_; };
    Tensor<T>  getOutputErrors() override { return dY_; };
    void dump() override {
//	std::cout<<"Successor:"<<std::endl;
	X_->dump();
	dX_->dump();
	holder_->dump();
    };



private:
    // хранилище данных и псевдонимы для тензоров
    typename DataHolder<T>::uPtr holder_;
    Tensor<T> X_;
    Tensor<T> dX_;
    Tensor<T> Y_;
    Tensor<T> dY_;
    ANN<T> *precursor_;
};

#endif