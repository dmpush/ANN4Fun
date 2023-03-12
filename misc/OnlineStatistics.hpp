#ifndef __ONLINE_STATISTICS_HPP__
#define __ONLINE_STATISTICS_HPP__

#include <iostream>
#include <chrono>
#include <cmath>
#include <cassert>
/**
    @brief OnlineStatistics -- класс для онлайновой оценки статистических параметров последовательностей.
*/
template<typename T=double>
class OnlineStatistics {
public:
private:
    double mean_, mean2_;
    size_t counter_;
public:
    OnlineStatistics() :   mean_(0), mean2_(0), counter_(0) {};
    /// @brief добавляет элемент последовательности.
    void  update(T val) {
	mean_+=val;
	mean2_+=val*val;
	counter_++;
    };
    /// @brief сброс внутреннего состояния. Отмечает начало последовательности.
    void cleanup() {
	mean_=0.0;
	mean2_=0.0;
	counter_=0;
    };
    /// @brief Функция измерения среднего значения последовательности.
    /// @returns матожидание последовательности.
    T getMean() {
	assert(counter_>0);
	return static_cast<T>(mean_)/static_cast<T>(counter_);
    };
    /// @brief Функция измерения стандартного отклонения последовательности.
    /// @returns СКО последовательности.
    T getStd() {
	assert(counter_>0);
	double M=static_cast<double>(mean_)/static_cast<double>(counter_);
	double M2=static_cast<double>(mean2_)/static_cast<double>(counter_);
	return static_cast<T>(std::sqrt(std::abs(M*M-M2)));
    };
};



#endif