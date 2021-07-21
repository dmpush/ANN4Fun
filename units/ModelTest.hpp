#ifndef __MODEL_TEST_HPP_
#define __MODEL_TEST_HPP_

#include <iostream>
#include <string>
#include <cassert>
#include <cmath> //abs
#include <random>
#include <algorithm> //min,max
#include <memory>
#include <stdexcept>

#include <DataHolder.hpp>
#include <Model.hpp>
#include <Dropout.hpp>
#include <Timer.hpp>
#include <Console.hpp>
using namespace std;
/**
    @brief Абстрактный класс для тестирования ИНС разных архитектур. Инкапсулирует обучение и тестирование.
    Дочерние классы должны конкретизировать обучающую выборку, условия прохождения теста и модель сети.
*/

template<typename T>
class ModelTest {
public:
    ModelTest(size_t numBatches=100, size_t batchSize=10, bool enabledDropout=false) :
	seed_{},
	rdev_{seed_()},
	enabledDropout_(enabledDropout),
	batch_size_(batchSize),
	num_batches_(numBatches),
	uniform_(0.0, 1.0),
	normal_(0.0, 1.0),
	elps_fwd_{std::make_shared<Timer<double>>()},
	elps_bwd_{std::make_shared<Timer<double>>()},
	elps_train_{std::make_shared<Timer<double>>()} {
	};
    ~ModelTest() = default;
    /// @brief Установить размер батча.
    /// @param sz размер батча.
    ModelTest* setBatchSize(size_t sz) {
	batch_size_=sz;
	return this; 
    };
    /// @brief Установить число батчей.
    /// @param sz размер батча.
    ModelTest* setNumBatches(size_t sz) {
	num_batches_=sz;
	return this;
    };
    /// @brief Пользовательский обработчик, вызывается для моделей по завершении тестирования.
    /// Предназначен для операций вроде формирования лога, записи модели/данных в файл и т.д.
    /// @param model модель;
    virtual void onTunedModel( typename Model<T>::sPtr) {};
    /// @brief виртуальный метод-конструктор создания модели ИНС.
    virtual typename Model<T>::sPtr buildModel() = 0;
    /// @brief виртуальный предикат - условие прохождения теста модели.
    virtual bool assertion() { return error_max_<0.1; };
    /// @brief метод генерации входов для обучающих примеров - семплирование.
    virtual std::vector<T> getInput()  {
	std::vector<T> out(2);
	for(auto& it: out) {
	    it=static_cast<T>(uniformNoise()*2.0-1);
	};
	return out;
    };
    /// @brief метод генерации выходов обучающих примеров по входам.
    virtual std::vector<T> getOutput(const std::vector<T> in) = 0;
    /// @brief
    /// @returns
    virtual size_t getNumInputs() = 0;
    /// @brief
    /// @returns
    virtual size_t getNumOutputs() = 0;
    /// @brief Прогоны тестирования.
    /// @param repeats  число прогонов теста.
    size_t run(size_t repeats=100, bool verbose=true) {
	elps_fwd_->cleanup();
	elps_bwd_->cleanup();
	elps_train_->cleanup();
	size_t cnt=0;
	for(size_t k=0; k<repeats; k++)
	    cnt += step() ? 1u : 0u;
	if(verbose) {
	    elapsed();
	    float per=100.0f * static_cast<float>(cnt) / static_cast<float>(repeats);
	    cout<<" ";
	    if(per > 90.0)
		cout<<Console().fgColor(Console::aqua);
	    else if(per > 50.0)
		cout<<Console().fgColor(Console::green);
	    else if(per > 10.0)
		cout<<Console().fgColor(Console::yellow);
	    else
		cout<<Console().fgColor(Console::red);
	    cout<<cnt<<Console().clear()<<" из "<<Console().fgColor(Console::aqua)<<repeats<<Console().clear()<<" ";
	};
	return cnt;
    };

    /// @brief Максимальная по модулю ошибка.
    /// @returns максимальную по модулю ошибку обучения 
    auto getErrorMax() { return error_max_; };
    /// @brief СКО.
    /// @returns среднеквадратическую ошибку обучения
    auto getErrorMeanSquare() { return error_mse_; };
    /// @brief Псевдослучайный генератор действительных чисел из диапазона [0..1] с равномерным распределением.
    /// @returns псевдослучайное число.
    auto uniformNoise() { return uniform_(rdev_); };
    /// @brief Псевдослучайный генератор нормального распределения.
    /// @returns псевдослучайные числа с нормальным распределением вероятностей.
    auto gaussianNoise() { return normal_(rdev_); };

public:
    void elapsed() {
    cout<<Console().fgColor(Console::green)<<
    " ("<<*(elps_fwd_.get())<<"/"<<*(elps_bwd_.get())<<"/"<<*(elps_train_.get())<<") "<<
    Console().clear();
    };
    void enableDropout() { enabledDropout_=true; };
    void disableDropout() { enabledDropout_=false; };
private:
    /// @brief Один прогон обучения и тестирования модели.
    /// @returns true, если модель успешно прошла тестирование.
    bool step() {
	try {
	    typename Dropout<T>::Enabled dropout_on(true);
	    typename Dropout<T>::Enabled dropout_off(false);
	    typename Dropout<T>::Update  dropout_update;
	    auto model=buildModel();
	    assert(model->getNumInputs() == getNumInputs());
	    assert(model->getNumOutputs() == getNumOutputs());
	    if(enabledDropout_)
		model->notify(&dropout_on);
	    for(size_t i=0; i<num_batches_; i++) {
		if(enabledDropout_)
		    model->notify(&dropout_update);
		model->batchBegin();
		for(size_t k=0; k<batch_size_; k++) {
		    auto inputs=getInput();
		    auto outputs=getOutput(inputs);

		    for(size_t p=0; p<inputs.size(); p++)
			model->setInput(p, inputs[p]);
		    elps_fwd_->tic();
		    model->forward();
		    elps_fwd_->toc();
		    for(size_t p=0; p<outputs.size(); p++)
			model->setOutput(p, outputs[p]); //AND
		    elps_bwd_->tic();
		    model->backward();
		    elps_bwd_->toc();
		};
		elps_train_->tic();
		model->batchEnd();
		elps_train_->toc();
	    };
	    // проверка
	    if(enabledDropout_)
		model->notify(&dropout_off);
	    error_max_=0;
	    error_mse_=0;
	    for(size_t k=0; k<100; k++) {
		auto inputs=getInput();
		auto outputs=getOutput(inputs);

		for(size_t p=0; p<inputs.size(); p++)
		    model->setInput(p, inputs[p]);
		model->forward();

		std::vector<double> errors(outputs.size());
		for(size_t p=0; p<outputs.size(); p++)
		    errors[p] = std::abs(outputs[p] - model->getOutput(p));
		for(auto err: errors) {
		    error_max_=std::max(error_max_, err);
		    error_mse_ += err*err/100.0/static_cast<double>(errors.size());
		};
	    }; //for
//	    std::cout<<error_mse_<<endl;
	    onTunedModel(model);
	    if(!assertion())
		return false;
	} catch(std::runtime_error const &) {
	    return false;
	};
	return true;
    }; // step

protected:
    std::random_device seed_;
    std::mt19937 rdev_;
private:
    bool enabledDropout_;
    double error_max_;
    double error_mse_;
    size_t batch_size_;
    size_t num_batches_;
    std::uniform_real_distribution<double> uniform_;
    std::normal_distribution<double> normal_;
    std::shared_ptr<Timer<double>> elps_fwd_;
    std::shared_ptr<Timer<double>> elps_bwd_;
    std::shared_ptr<Timer<double>> elps_train_;
};

#endif
