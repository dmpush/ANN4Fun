#ifndef __SUCCESSOR_HPP__
#define __SUCCESSOR_HPP__

#include <iostream>
#include <vector>
#include <stdexcept>
#include <ANN.hpp>
#include <IDataHolder.hpp>
#include <IBackendFactory.hpp>
#include <Succession.hpp>

/**
    @brief Successor - сеть, которая явяется очередным слоем в составе другой сети. Имеет собственную память - 
    выходные значения и сигналы ошибок.
*/
template<typename T>
class Successor : public Succession<T> {
public:
    Successor(ANN<T> *ann, std::vector<size_t> Nout) : Succession<T>(ann), 
	precursor_(ann), 
	output_shape_(Nout),
	holder_{nullptr},
	X_{nullptr},
	dX_{nullptr},
	Y_{nullptr},
	dY_{nullptr}
    {
    };

    Successor(ANN<T> *ann) : Successor(ann, ann->shape()) {};

    virtual ~Successor() = default;

    void build(typename IBackendFactory<T>::sPtr factory) override {
	// сеть является владельцем своих входов и выходов
	holder_=std::move(factory->makeHolderU());
	holder_->append("Y", output_shape_);
	holder_->append("dY", output_shape_);
	holder_->build();
	// псевдонимы
	X_ = precursor_ -> getOutputs();
	dX_= precursor_ -> getOutputErrors();
	Y_ = holder_    -> get("Y");
	dY_= holder_    -> get("dY");
	holder_->fill(T(0));
    };




    TensorPtr<T>  getInputs()  override  { return X_; };
    TensorPtr<T>  getOutputs() override  { return Y_; };
    TensorPtr<T>  getInputErrors()  override { return dX_; };
    TensorPtr<T>  getOutputErrors() override { return dY_; };
    void dump() override {
//	std::cout<<"Successor:"<<std::endl;
	X_->dump();
	dX_->dump();
	holder_->dump();
    };
    ANN<T>* getPrecursor() { return precursor_; };
    std::vector<size_t> shape() override { return output_shape_; };


private:
    // хранилище данных и псевдонимы для тензоров
    ANN<T> *precursor_;
    const std::vector<size_t> output_shape_;
    typename IDataHolder<T>::uPtr holder_;
    TensorPtr<T> X_;
    TensorPtr<T> dX_;
    TensorPtr<T> Y_;
    TensorPtr<T> dY_;
};

#endif