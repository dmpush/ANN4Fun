#ifndef __SKIP_CONNECTIONS__
#define __SKIP_CONNECTIONS__

#include <iostream>
#include <vector>
#include <stdexcept>
#include <ANN.hpp>
#include <AbstractTutor.hpp>
#include <IBackendFactory.hpp>
#include <Succession.hpp>
/**
 * @brief Класс реализует кросс-соединения идущие параллельно внутренней сети.
 * 
 */
template <typename T>
class SkipConnections : public Succession<T>
{
public:
    using sPtr = std::shared_ptr<SkipConnections<T>>;
    explicit SkipConnections(typename ANN<T>::sPtr ann) : Succession<T>(ann), 
        ann_(ann),
        X_{ann->getInputs()},
        Y_{ann->getOutputs()},
        dX_{ann->getInputErrors()},
        dY_{ann->getOutputErrors()} {
            assert(X_->dim() == Y_->dim());
            assert(X_->dims() == Y_->dims());
        };
    virtual ~SkipConnections(){};
    void build(typename IBackendFactory<T>::sPtr) override final{};
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
private:
    const typename ANN<T>::sPtr ann_;
    const typename ITensor<T>::sPtr X_, Y_, dX_, dY_;
};

#endif