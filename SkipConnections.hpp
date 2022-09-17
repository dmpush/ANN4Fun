#ifndef __SKIP_CONNECTIONS__
#define __SKIP_CONNECTIONS__

#include <iostream>
#include <vector>
#include <stdexcept>
#include <ANN.hpp>
#include <AbstractTutor.hpp>
#include <IBackendFactory.hpp>
#include <Succession.hpp>
#include <Model.hpp>

/**
 * @brief Класс реализует кросс-соединения идущие параллельно внутренней сети.
 * 
 */
template <typename T>
class SkipConnections : public Succession<T>
{
public:
    using sPtr = std::shared_ptr<SkipConnections<T>>;
    explicit SkipConnections(typename Model<T>::sPtr ann) : Succession<T>(ann), 
        ann_{ann},
        X_{nullptr},
        Y_{nullptr},
        dX_{nullptr},
        dY_{nullptr} {
        };
   explicit SkipConnections(const std::vector<size_t>& inputShape) : SkipConnections( std::make_shared<Model<T>>(inputShape) ) {};
/* 
   explicit SkipConnections(const std::vector<size_t>& inputShape) : Succession<T>(ann), 
        ann_{std::make_shared<Model<T>>(inputShape)},
        X_{nullptr},
        Y_{nullptr},
        dX_{nullptr},
        dY_{nullptr} {
        };
*/
    virtual ~SkipConnections(){};
    void build(typename IBackendFactory<T>::sPtr factory) override final{
	ann_->build(factory);
	X_=ann_->getInputs();
	Y_=ann_->getOutputs();
	dX_=ann_->getInputErrors();
	dY_=ann_->getOutputErrors();
	assert(X_->dim() == Y_->dim());
	assert(X_->dims() == Y_->dims());
    };
    void setTutor(typename AbstractTutor<T>::uPtr) override final{};
    TensorPtr<T> getInputs() override { return X_; };
    TensorPtr<T> getOutputs() override { return Y_; };
    TensorPtr<T> getInputErrors() override { return dX_; };
    TensorPtr<T> getOutputErrors() override { return dY_; };

    void forward() override final {
        assert(ann_);
        assert(X_);
        assert(Y_);
        ann_->forward();
        Y_->append(X_);
    };
    void backward() override final {
	    assert(ann_);
        assert(dX_);
        assert(dY_);
        ann_->backward();
        dX_->append(dY_);
    };
    void batchBegin() override final{
	    assert(ann_);
	    ann_->batchBegin();
    };
    void batchEnd() override final {
	    assert(ann_);
	    ann_->batchEnd();
    };
    void notify(typename ANN<T>::Notification* notice) override final {
	    assert(ann_);
	    ann_->notify(notice);
    };
    void dump() override {
	    ann_->dump();
    };
    std::vector<size_t> shape() override { return ann_->shape(); };
    auto getModel() { return ann_; };
private:
    typename Model<T>::sPtr ann_;
    typename ITensor<T>::sPtr X_, Y_, dX_, dY_;
};

#endif
