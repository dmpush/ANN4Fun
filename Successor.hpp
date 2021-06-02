#ifndef __SUCCESSOR_HPP__
#define __SUCCESSOR_HPP__

#include <vector>
#include <stdexcept>
#include <ANN.hpp>
#include <DataHolder.hpp>

template<typename T>
class Successor : public ANN<T> {
public:
    Successor(ANN<T> *ann, std::vector<size_t> Nout) : precursor_(ann), ANN<T>() {
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
	holder_->description();
    };
    virtual ~Successor() = default;




    Tensor<T>  getInputs()  override  { return X_; };
    Tensor<T>  getOutputs() override  { return Y_; };
    Tensor<T>  getInputErrors()  override { return dX_; };
    Tensor<T>  getOutputErrors() override { return dY_; };

    T getOutput(size_t index)          override { return Y_->get(index); };
    T setOutput(size_t index, T value) override { return dY_->set(index, value - Y_->get(index)); };

    T getInput(size_t index)          override { return X_->get(index); };
    T setInput(size_t index, T value) override { return X_->set(index, value); };

    T setError(size_t index, T value)    override { return dY_->set(index, value); };
    T appendError(size_t index, T value) override { return dY_->set(index, dY_->get(index) + value); };


    size_t getNumInputs()  override { return X_->size(); };
    size_t getNumOutputs() override { return Y_->size(); };

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