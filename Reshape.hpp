#ifndef __RESHAPE_HPP__
#define __RESHAPE_HPP__

#include <stdexcept>
#include <string>
#include <memory>
#include <vector>

#include <ANN.hpp>
#include <Successor.hpp>
#include <IBackendFactory.hpp>
#include <IDataHolder.hpp>
#include <AbstractTutor.hpp>

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
    explicit Reshape(ANN<T>* ann, const std::vector<size_t>& newShape) :
	Successor<T>(ann, newShape),
	X_{nullptr},
	Y_{nullptr}, 
	dX_{nullptr},
	dY_{nullptr} {
	auto oldShape=ann->shape();
	size_t sz1=1;
	for(auto i: newShape)
	    sz1*=i;
	size_t sz2=1;
	for(auto i: oldShape)
	    sz2*=i;
	if(sz1 != sz2)
	    throw std::runtime_error("Операция Reshape невозможна: требуемый объем не сопряжен с сетью.");
    };
    ~Reshape() = default;
    void build(typename IBackendFactory<T>::sPtr factory) override {
	Successor<T>::build(factory);
	X_=Successor<T>::getInputs();
	dX_=Successor<T>::getInputErrors();
	holder_=factory->makeHolderU();
	holder_->append("Y", Successor<T>::shape());
	holder_->append("dY", Successor<T>::shape());
	holder_->build();
	Y_=holder_->get("Y");
    	dY_=holder_->get("dY");
    };


    void forward() override {
	assert(X_);
	assert(Y_);
	Y_->copy(X_);
    };
    void backward() override {
	assert(dX_);
	assert(dY_);
	dX_->copy(dY_);
    };
    void batchBegin() override {
	assert(X_);
	assert(Y_);
	assert(dX_);
	assert(dY_);
    };
    void batchEnd() override {
    };
    void setTutor(typename AbstractTutor<T>::uPtr) override {
    };

    TensorPtr<T> getOutputs()      override { return Y_; };
    TensorPtr<T> getOutputErrors() override { return dY_; };

private:
    typename IDataHolder<T>::uPtr holder_;
    TensorPtr<T> X_, Y_;
    TensorPtr<T> dX_, dY_;
protected:
};

#endif