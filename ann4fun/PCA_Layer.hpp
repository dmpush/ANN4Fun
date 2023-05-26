#ifndef __PCA_LAYER_HPP__
#define __PCA_LAYER_HPP__

#include <ANN.hpp>
#include <Succession.hpp>
#include <Layer.hpp>
#include <iostream>
#include <vector>

template <typename T>
class PCA_Layer : public Succession<T> {
public:
  explicit PCA_Layer(typename ANN<T>::sPtr previos, const std::vector<size_t>& Sout, size_t Nint) : 
    Succession<T>(previos),
    first_{std::make_shared<Layer<T>>(previos, std::vector<size_t>({Nint})) },
    second_{std::make_shared<Layer<T>>(first_, Sout)} {

  }
  virtual ~PCA_Layer() = default;
  void build(typename IBackendFactory<T>::sPtr factory) override final {
    first_->build(factory);
    second_->build(factory);
  /*
    auto inputShape=getInputs()->dims();
    auto outputShape=getOutputs()->dims();
    auto internalShape=first_->dims();
    assert(inputShape.size()==1);
    assert(outputShape.size()==1);
    assert(internalShape.size()==1);
    auto Nin=inputShape[0];
    auto Nout=outputShape[0];
    auto Nint=internalShape[0];
    if( (Nin+Nout)*Nint >= Nin*Nout) {
      std::cout
        <<"ПРЕДУПРЕЖДЕНИЕ PCA_Layer : скрытая размерность должна быть в диапазоне от 1 до "
        <<static_cast<float>(Nin * Nout) / static_cast<float>(Nin + Nout)
        <<std::endl;
    }    
  */  
  };

  void setTutor(typename AbstractTutor<T>::uPtr tutor) override final {
    first_->setTutor(std::move(tutor->clone()));
    second_->setTutor(std::move(tutor->clone()));
  };  
    void forward() override final{
	  assert(first_);
	  assert(second_);
	  first_->forward();
	  second_->getInputs()->copy( first_->getOutputs());
	  second_->forward();
  };  
  void backward() override final {
	  assert(first_);
	  assert(second_);
	  second_->backward();
	  first_->getOutputErrors()->copy(second_->getInputErrors());
	  first_->backward();
  }
  void batchBegin() override final{
	  assert(first_);
	  assert(second_);
	  first_->batchBegin();
	  second_->batchBegin();
  };
  void batchEnd() override final {
	  assert(first_);
	  assert(second_);
	  first_->batchEnd();
	  second_->batchEnd();
  };  
  // этот слой работает как единое целое
  void notify(typename ANN<T>::Notification* notice) override final {};
  TensorPtr<T>  getInputs()  override  { return first_->getInputs(); };
  TensorPtr<T>  getOutputs() override  { return second_->getOutputs(); };
  TensorPtr<T>  getInputErrors()  override { return first_->getInputErrors(); };
  TensorPtr<T>  getOutputErrors() override { return second_->getOutputErrors(); };

  void dump() override {
	  first_->dump();
	  second_->dump();
  };

  std::vector<size_t> shape() override { return second_->shape(); };
private:
  typename Layer<T>::sPtr first_, second_;
};

#endif