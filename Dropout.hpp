#ifndef __DROPOUT_HPP__
#define __DROPOUT_HPP__

#include <stdexcept>
#include <string>
#include <memory>
#include <vector>

#include <ANN.hpp>
#include <Successor.hpp>
#include <DataHolder.hpp>
#include <ITensor.hpp>
#include <AbstractTutor.hpp>

/** 
    @brief Dropout - Слой дропаута нейронной сети.
*/
template<typename T=float>
class Dropout : public Successor<T> {
public:
    Dropout() = delete;
    Dropout(const Dropout&) = delete;
    explicit Dropout(ANN<T>* ann, double probability) : Successor<T>(ann), probability_(1.0-probability), enabled_(true)  {
	X_=Successor<T>::getInputs();
	Y_=Successor<T>::getOutputs();
	dX_=Successor<T>::getInputErrors();
	dY_=Successor<T>::getOutputErrors();
	holder_=std::make_unique<DataHolder<T>>();
	holder_->append("conduction", X_->dims());
	holder_->build();
	conduction_=holder_->get("conduction");
	update();
    };
    ~Dropout() = default;


    void forward() override {
	if(enabled_)
	    Y_->prod(X_, conduction_);
	else
	    Y_->copy(X_);
    };
    void backward() override {
	if(enabled_)
	    dX_->prod(dY_, conduction_);
	else
	    dX_->copy(dY_);
    };
    void batchBegin() override {
    };
    void batchEnd() override {
    };
    void setTutor(typename AbstractTutor<T>::uPtr) override {
    };
    /// @brief Управляющее сообщение для слоя Dropout -- включение/выключение дропаута.
    class Enabled: public ANN<T>::Notification {
    public:
	Enabled(bool enabled) :  ANN<T>::Notification(), enabled_(enabled) {};
	bool isEnabled() { return enabled_; };
    private:
	bool enabled_;
    };
    /// @brief Управляющее сообщение для слоя Dropout -- семплирование выключенных нейронов.
    class Update: public ANN<T>::Notification {
    public:
	Update() : ANN<T>::Notification() {}
    };
    /// @brief Управление слоем Dropout.
    /// @todo Этот код просто ужасен. Придумать, как его написать в нормальном ООП.
    /// @param notice - управляющее сообщение.
    void notify(typename ANN<T>::Notification* notice) override {
	auto en=dynamic_cast<typename Dropout<T>::Enabled*>(notice);
	if(en)
	    enabled_= en->isEnabled();
	else {
	    auto updt=dynamic_cast<typename Dropout<T>::Update*>(notice);
	    if(updt)
		update();
	}
    };
private:
    /// @brief Обновление множеств активных/неактивных нейронов.
    void update() { // множитель нужен для сохранения матожидания 
	conduction_->bernoulliNoise(probability_, 1.0/probability_, 0.0);
    };
private:
    TensorPtr<T> X_, Y_;
    TensorPtr<T> dX_, dY_;
    double probability_;
    typename IDataHolder<T>::uPtr holder_;
    TensorPtr<T> conduction_;
    bool enabled_;
protected:
};

#endif