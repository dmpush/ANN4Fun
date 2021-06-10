#ifndef __RESHAPE_HPP__
#define __RESHAPE_HPP__

#include <stdexcept>
#include <string>
#include <memory>
#include <vector>

#include <ANN.hpp>
#include <Successor.hpp>
#include <DataHolder.hpp>
#include <AbstractTutor.hpp>
#include <TensorMath.hpp>

/** 
    @brief Reshape - преобразование размерностей тензоров между слоями. 
    Должно быть выполено условие вход.size()==выход.size()
*/
template<typename T>
class Reshape : public Successor<T> {
public:
    Reshape() = delete;
    Reshape(const Reshape&) = delete;
    /// @param ann - входной слой сети
    /// @param shape - размерности выходного тензора
    explicit Reshape(ANN<T>* ann, const std::vector<size_t>& shape) : Successor<T>(ann) {
	size_t sz=1;
	for(auto i: shape)
	    sz*=i;
	X_=Successor<T>::getInputs();
	dX_=Successor<T>::getInputErrors();
	if(sz != X_->size())
	    throw std::runtime_error("Операция Reshape невозможна: требуемый объем не сопряжен с сетью.");
	holder_=std::make_unique<DataHolder<T>>();
	holder_->append("Y", shape);
	holder_->append("dY", shape);
	holder_->build();
	Y_=holder_->get("Y");
	dY_=holder_->get("dY");
    };
    ~Reshape() = default;

    void forward() override {
	tensormath::copy<T>(X_, Y_);
    };
    void backward() override {
	tensormath::copy<T>(dY_, dX_);
    };
    void batchBegin() override {
    };
    void batchEnd() override {
    };
    void setTutor(typename AbstractTutor<T>::uPtr) override {
    };

    Tensor<T> getOutputs()      override { return Y_; };
    Tensor<T> getOutputErrors() override { return dY_; };

private:
    typename DataHolder<T>::uPtr holder_;
    Tensor<T> X_, Y_;
    Tensor<T> dX_, dY_;
protected:
};

#endif