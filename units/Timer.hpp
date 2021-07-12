#ifndef __ELAPSED_TIMER_HPP__
#define __ELAPSED_TIMER_HPP__

#include <iostream>
#include <chrono>
#include <cmath>
#include <cassert>
#include <OnlineStatistics.hpp>
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
    OnlineStatistics<T> stat;
public:
    Timer() :  ts_(clock_t::now()), stat{} {};
    /// @brief Засекает начало интервала времени.
    void tic() {
	ts_=clock_t::now();
    };
    /// @brief Засекает конец интервала времени.
    /// @returns длительность интервала времени.
    T  toc() {
	auto dur = std::chrono::duration_cast<second_t>(clock_t::now() - ts_).count();
	stat.update(dur);
	return dur;
    };
    /// @brief Засекает начало серии испытаний.
    void cleanup() {
	stat.cleanup();
    };
    /// @brief Функция измерения среднего значение интервала в серии испытаний.
    /// @returns средняя длительность интервала времени.
    T getMean() {
	return stat.getMean();
    };
    /// @brief Функция измерения стандартного отклонения длительности интервала в серии испытаний.
    /// @returns стандартное отклонение длительности интервала времени.
    T getStd() {
	return stat.getStd();
    };

};



#endif