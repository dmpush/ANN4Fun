#ifndef __ELAPSED_TIMER_HPP__
#define __ELAPSED_TIMER_HPP__

#include <iostream>
#include <chrono>
#include <cmath>
#include <cassert>
/**
    @brief Timer класс для измерения таймингов между участками кода.
    Поддерживает измерения времени при серийных испытаниях.
*/
template<typename T=double>
class Timer {
public:
private:
    using clock_t=std::chrono::high_resolution_clock;
    using second_t=std::chrono::duration<T, std::ratio<1>>;
    std::chrono::time_point<clock_t> ts_;
    double mean_, mean2_;
    size_t counter_;
public:
    Timer() :  ts_(clock_t::now()), mean_(0), mean2_(0), counter_(0) {};
    /// @brief Засекает начало интервала времени.
    void tic() {
	ts_=clock_t::now();
    };
    /// @brief Засекает конец интервала времени.
    /// @returns длительность интервала времени.
    T  toc() {
	auto dur = std::chrono::duration_cast<second_t>(clock_t::now() - ts_).count();
	mean_+=dur;
	mean2_+=dur*dur;
	counter_++;
	return dur;
    };
    /// @brief Засекает начало серии испытаний.
    void cleanup() {
	mean_=0.0;
	mean2_=0.0;
	counter_=0;
    };
    /// @brief Функция измерения среднего значение интервала в серии испытаний.
    /// @returns средняя длительность интервала времени.
    T getMean() {
	assert(counter_>0);
	return static_cast<T>(mean_)/static_cast<T>(counter_);
    };
    /// @brief Функция измерения стандартного отклонения длительности интервала в серии испытаний.
    /// @returns стандартное отклонение длительности интервала времени.
    T getStd() {
	assert(counter_>0);
	double M=static_cast<double>(mean_)/static_cast<double>(counter_);
	double M2=static_cast<double>(mean2_)/static_cast<double>(counter_);
	return static_cast<T>(std::sqrt(std::abs(M*M-M2)));
    };

};



#endif