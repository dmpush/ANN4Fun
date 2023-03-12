#ifndef __COMPOSITION_HPP__
#define __COMPOSITION_HPP__

#include <iostream>
#include <vector>
#include <stdexcept>
#include <ANN.hpp>
#include <AbstractTutor.hpp>
#include <IBackendFactory.hpp>
#include <Succession.hpp>

/**
    @brief Композиция (последовательное соединение) двух независимых сетей. Для создания GAN, автоэнкодеров и т.д.
*/
template<typename T>
class Composition : public Succession<T> {
public:
    using sPtr=std::shared_ptr<Composition<T>>;
    explicit Composition(typename ANN<T>::sPtr first, typename ANN<T>::sPtr second) : Succession<T>(first), 
	first_(first),
	second_(second)
    {
    };

    virtual ~Composition() = default;

    void build(typename IBackendFactory<T>::sPtr) override final {
    };

    void setTutor(typename AbstractTutor<T>::uPtr) override final {
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
    void notify(ANN<T>::Notification* notice) override final {
	assert(first_);
	assert(second_);
	first_->notify(notice);
	second_->notify(notice);
    };


    TensorPtr<T>  getInputs()  override  { return first_->getInputs(); };
    TensorPtr<T>  getOutputs() override  { return second_->getOutputs(); };
    TensorPtr<T>  getInputErrors()  override { return first_->getInputErrors(); };
    TensorPtr<T>  getOutputErrors() override { return second_->getOutputErrors(); };
    void dump() override {
	first_->dump();
	second_->dump();
    };
    /// @brief первая подсеть
    typename ANN<T>::sPtr getFirst() const { return first_; };
    /// @brief вторая подсеть
    typename ANN<T>::sPtr getSecond() const { return second_; };

    std::vector<size_t> shape() override { return second_->shape(); };

private:
    const typename ANN<T>::sPtr first_, second_;
};

#endif